"""Algorithms to optimize model parameters for easier recovery."""

import torch
import time


from ...common import optimizer_lookup

from .objectives import DeepLayerRatioMatching
from ...analysis.metrics import psnr_compute

class RecoveryOptimizer():
    """Implemented Objectives:
        *  deep-layer-ratio-matching

        Todo:
        * Pixelwise-grad Recovery
        * Pixelwise sub-bit recovery [Recover image pixel bit if grad_value somewhere > 0]
        * Meta-learning solution
        * A learned encoder-decoder
        * Gradient-Matching (like in WB) solution

    Also todo:
        * filter-based recovery addon [recovery from dct coefficients?]
        * Fake synthetic data
    """
    def __init__(self, model, loss, cfg_data, cfg_impl, cfg_optim,
                 setup=dict(dtype=torch.float, device=torch.device('cpu')), external_data=None):
        """Initialize with info from the server. Data could be optional in the future."""
        self.model = model.to(**setup)

        self.model.train()
        self.loss = loss
        self.setup = setup

        self.cfg_data = cfg_data
        self.cfg_optim = cfg_optim
        self.cfg_impl = cfg_impl

        self.dm = torch.as_tensor(cfg_data.mean, **setup)[None, :, None, None]
        self.ds = torch.as_tensor(cfg_data.std, **setup)[None, :, None, None]

        self.dataloader = external_data
        self.feature_shapes = self._introspect_model()

        if self.cfg_optim.objective == 'deep-layer-ratio-matching':
            self.objective = DeepLayerRatioMatching(model, loss, cfg_optim.target_shape, cfg_optim.layers)
        elif self.cfg_optim.objective == 'pixel-matching':
            self.objective = PixelMatching(model, loss, cfg_optim.target_shape)
        else:
            raise ValueError(f'Invalid objective {self.cfg_optim.objective} given.')
        self.effective_batch_size = self.objective.target_shape[0]

    def _introspect_model(self):
        """Compute model feature shapes."""
        feature_shapes = dict()

        def named_hook(name):
            def hook_fn(module, input, output):
                feature_shapes[name] = dict(shape=input[0].shape, info=str(module))
            return hook_fn

        hooks_list = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks_list.append(module.register_forward_hook(named_hook(name)))

        throughput = torch.zeros([1, *self.cfg_data.shape], **self.setup)
        self.model(throughput)
        # True user data is returned only for analysis
        [h.remove() for h in hooks_list]
        return feature_shapes



    def optimize_recovery(self):
        """Run an optimization-based algorithm to minimize the target objective over the given real or synth. data."""

        optimizer, scheduler = optimizer_lookup(self.model.parameters(), **self.cfg_optim.optim)
        num_blocks = len(self.dataloader)
        for iteration in range(self.cfg_optim.optim.max_iterations):
            step_final_loss, step_default_loss, step_psnr = 0, 0, 0
            time_stamp = time.time()
            for block, (inputs, labels) in enumerate(self.dataloader):
                optimizer.zero_grad()
                chunks_in_block = max(labels.shape[0] // self.effective_batch_size, 1)

                inputs = inputs.to(**self.setup, non_blocking=self.cfg_impl.non_blocking)
                labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=self.cfg_impl.non_blocking)

                input_chunks = torch.split(inputs, self.effective_batch_size, dim=0)[:chunks_in_block]
                label_chunks = torch.split(labels, self.effective_batch_size, dim=0)[:chunks_in_block]

                for sub_idx, (input_chunk, label_chunk) in enumerate(zip(input_chunks, label_chunks)):

                    outputs, final_loss, default_loss = self.objective(input_chunk, label_chunk)
                    final_loss.backward()
                    # [p.grad.sign() for p in self.model.parameters()]
                    optimizer.step()
                    with torch.no_grad():
                        inputs_candidate = outputs.detach().mul(self.ds).add(self.dm).clamp_(0, 1)
                        reference_candidate = input_chunk.detach().mul(self.ds).add(self.dm).clamp_(0, 1)
                        psnr = psnr_compute(inputs_candidate, reference_candidate, batched=False, factor=1.0)
                        step_psnr += psnr
                        step_final_loss += final_loss.detach()
                        step_default_loss += default_loss.detach()

                        if not final_loss.isfinite():
                            raise ValueError('Nonfinite values introduced in param optimization!')
                print(f'Block: {block} | Time: {time.time() - time_stamp:4.2f}|Obj:{final_loss.item():7.4f}|PSNR:{psnr:4.2f}')
            print(f'|Iteration {iteration:<4} | Time: {time.time() - time_stamp:4.2f}s | '
                  f'Objective: {step_final_loss / num_blocks / chunks_in_block:7.4f} | '
                  f'Data Loss: {step_default_loss / num_blocks / chunks_in_block:7.4f} | '
                  f'PSNR: {step_psnr / num_blocks / chunks_in_block:4.2f} |')
            scheduler.step()
