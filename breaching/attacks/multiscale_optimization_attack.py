"""Implementation for gradient inversion attacks.

This covers complicates optimization-based reconstruction attacks by introducing multiscale recovery.
This is usually (in CV in general) helpful for large image structures. Might be a bad idea here though?
"""

import torch
import torch.nn.functional as F
import time

from .optimization_based_attack import OptimizationBasedAttacker

import logging

log = logging.getLogger(__name__)


class MultiScaleOptimizationAttacker(OptimizationBasedAttacker):
    """Implements a wide spectrum of optimization-based attacks."""

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, dryrun=False):
        """Run a single reconstruction trial."""
        self.exitsignal = False  # Only used for early exit from KeyboardInterrupt
        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data["metadata"]["local_hyperparams"])

        assert self.data_shape[1] == self.data_shape[2]  # For simplicity for now
        C = self.data_shape[0]

        if self.cfg.scale_pyramid == "linear":
            increment = self.data_shape[2] // self.cfg.num_stages
            scale_pyramid = torch.arange(increment, self.data_shape[2] + 1, increment)
        elif self.cfg.scale_pyramid == "log":
            scales = torch.as_tensor([self.data_shape[2] / (2 ** i) for i in range(self.cfg.num_stages - 1, -1, -1)])
            scale_pyramid = scales.round().int()
        elif self.cfg.scale_pyramid == "trivial":
            scale_pyramid = [self.data_shape[2] for _ in range(self.cfg.num_stages)]
        else:
            raise ValueError(f"Invalid scale pyramid {cfg.scale_pyramid}.")

        # Initialize candidate reconstruction data on lowest scale:
        scale = scale_pyramid[0]
        num_data_points = shared_data["metadata"]["num_data_points"]
        candidate = self._initialize_data([num_data_points, C, scale, scale]).detach()
        best_candidate = self._initialize_data([num_data_points, *self.data_shape]).detach()
        scale_params = dict(mode="bilinear", align_corners=False)

        for stage, scale in enumerate(scale_pyramid):
            log.info(f"| Now solving stage {stage + 1}/{self.cfg.num_stages} with scale {scale}:")
            # Upsample base to new resolution
            if self.cfg.resize == "focus":
                p = torch.div(scale, 2, rounding_mode="floor")
                background = self._initialize_data([num_data_points, C, scale, scale]).detach()
                scaled_var = F.interpolate(candidate, size=p, **scale_params)
                cx = torch.div(scale - p, 2, rounding_mode="floor")
                background[:, :, cx : cx + p, cx : cx + p] = scaled_var
                candidate = background
            else:
                candidate = F.interpolate(candidate, size=scale, **scale_params)
            candidate = candidate.detach().requires_grad_()
            # Run stage:
            stage_candidate = self._run_stage(candidate, rec_model, shared_data, labels, stats, trial, stage, dryrun)
            best_candidate = F.interpolate(stage_candidate, size=self.data_shape[2], **scale_params)
            if dryrun or self.exitsignal:
                break

        return best_candidate

    def _run_stage(self, candidate, rec_model, shared_data, labels, stats, trial, stage, dryrun=False):
        """Optimization with respect to base_candidate."""
        best_candidate = candidate.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        # Initialize optimizers
        optimizer, scheduler = self._init_optimizer([candidate])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective(candidate, labels, rec_model, optimizer, shared_data, iteration)
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate.detach().clone()

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    log.info(
                        f"| S: {stage} - It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                    )
                    current_wallclock = timestamp

                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())
                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration} of stage {stage}!")
            self.exitsignal = True
        return best_candidate.detach()
