from __future__ import annotations

import math

import torch


def quantize_gradient(
    grad: torch.Tensor,
    nbits: int = 4,
    block_size: int = 64,
    stochastic: bool = False,
    preserve_norm: bool = False,
) -> torch.Tensor:
    """Quantize gradient tensor with block-wise uniform quantization as an inversion defense.

    Discretizes continuous gradient vectors to disrupt gradient-matching objectives
    (e.g., InvertingGradients, DLG, GradInversion) while preserving directional signals.

    Parameters:
        grad: Floating-point gradient tensor.
        nbits: Quantization bit width (e.g. 4 for INT4, 8 for INT8).
        block_size: Size of independent scaling blocks.
        stochastic: If True, uses unbiased stochastic rounding instead of deterministic rounding.
        preserve_norm: If True, rescales reconstructed gradient to match original L2 norm.

    Returns:
        Reconstructed defended gradient tensor in original float dtype and shape.
    """
    if grad.numel() == 0:
        return grad.clone()

    orig_shape = grad.shape
    orig_dtype = grad.dtype
    orig_device = grad.device

    flat = grad.detach().to(torch.float32).reshape(-1)
    numel = flat.numel()

    if numel == 0:
        return grad.clone()

    orig_norm = torch.norm(flat, p=2) if preserve_norm else None

    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        padded = torch.nn.functional.pad(flat, (0, pad_len), mode="constant", value=0.0)
    else:
        padded = flat

    blocks = padded.view(-1, block_size)
    qmax = (1 << (nbits - 1)) - 1  # 7 for 4-bit
    qmin = -(1 << (nbits - 1))  # -8 for 4-bit

    max_vals = torch.max(torch.abs(blocks), dim=1, keepdim=True)[0]
    scales = torch.clamp(max_vals / float(qmax), min=1e-8)

    scaled_blocks = blocks / scales

    if stochastic:
        floor_val = torch.floor(scaled_blocks)
        prob = scaled_blocks - floor_val
        rand = torch.rand_like(prob)
        quantized = floor_val + (rand < prob).float()
    else:
        quantized = torch.round(scaled_blocks)

    quantized = torch.clamp(quantized, qmin, qmax)
    recon_blocks = quantized * scales
    recon_flat = recon_blocks.view(-1)[:numel]

    if preserve_norm and orig_norm is not None and orig_norm > 1e-8:
        recon_norm = torch.norm(recon_flat, p=2)
        if recon_norm > 1e-8:
            recon_flat = recon_flat * (orig_norm / recon_norm)

    return recon_flat.view(orig_shape).to(device=orig_device, dtype=orig_dtype)


def quantize_gradient_list(
    grads: list[torch.Tensor],
    nbits: int = 4,
    block_size: int = 64,
    stochastic: bool = False,
    preserve_norm: bool = False,
    in_place: bool = True,
) -> list[torch.Tensor]:
    """Apply block-wise quantization to a list of gradient tensors."""
    out_grads = []
    for g in grads:
        qg = quantize_gradient(
            g,
            nbits=nbits,
            block_size=block_size,
            stochastic=stochastic,
            preserve_norm=preserve_norm,
        )
        if in_place:
            g.copy_(qg)
            out_grads.append(g)
        else:
            out_grads.append(qg)
    return out_grads


def prune_gradient(
    grad: torch.Tensor,
    sparsity: float = 0.9,
    method: str = "topk",
) -> torch.Tensor:
    """Prune gradient tensor as a baseline defense against inversion attacks.

    Parameters:
        grad: Input gradient tensor.
        sparsity: Fraction of values to zero out (e.g. 0.9 = 90% pruned).
        method: 'topk' (retain largest magnitude entries) or 'random'.

    Returns:
        Pruned gradient tensor with zeros at pruned positions.
    """
    if sparsity <= 0.0 or grad.numel() == 0:
        return grad.clone()

    flat = grad.detach().clone().reshape(-1)
    k = max(1, round((1.0 - sparsity) * flat.numel()))

    if method == "topk":
        _, indices = torch.topk(torch.abs(flat), k=k, largest=True)
    elif method == "random":
        indices = torch.randperm(flat.numel(), device=grad.device)[:k]
    else:
        raise ValueError(f"Unknown pruning method: {method}")

    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[indices] = True
    pruned = torch.where(mask, flat, torch.zeros_like(flat))
    return pruned.view(grad.shape)


def prune_gradient_list(
    grads: list[torch.Tensor],
    sparsity: float = 0.9,
    method: str = "topk",
    in_place: bool = True,
) -> list[torch.Tensor]:
    """Apply gradient pruning to a list of gradient tensors."""
    out_grads = []
    for g in grads:
        pg = prune_gradient(g, sparsity=sparsity, method=method)
        if in_place:
            g.copy_(pg)
            out_grads.append(g)
        else:
            out_grads.append(pg)
    return out_grads


class QuantizationDefense:
    """Configurable gradient quantization defense wrapper for federated privacy benchmarking."""

    def __init__(
        self,
        nbits: int = 4,
        block_size: int = 64,
        stochastic: bool = False,
        preserve_norm: bool = False,
    ):
        self.nbits = nbits
        self.block_size = block_size
        self.stochastic = stochastic
        self.preserve_norm = preserve_norm

    def __call__(
        self, grads: torch.Tensor | list[torch.Tensor], in_place: bool = True
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(grads, torch.Tensor):
            qg = quantize_gradient(
                grads,
                nbits=self.nbits,
                block_size=self.block_size,
                stochastic=self.stochastic,
                preserve_norm=self.preserve_norm,
            )
            if in_place:
                grads.copy_(qg)
                return grads
            return qg
        elif isinstance(grads, list):
            return quantize_gradient_list(
                grads,
                nbits=self.nbits,
                block_size=self.block_size,
                stochastic=self.stochastic,
                preserve_norm=self.preserve_norm,
                in_place=in_place,
            )
        else:
            raise TypeError(f"Unsupported grads type: {type(grads)}")

    def calc_compression_stats(
        self, grads: torch.Tensor | list[torch.Tensor]
    ) -> dict[str, float]:
        """Compute the communication wire size and theoretical compression savings."""
        if isinstance(grads, torch.Tensor):
            grads = [grads]

        orig_bytes = sum(g.numel() * g.element_size() for g in grads)
        quant_bytes = 0

        for g in grads:
            numel = g.numel()
            num_blocks = math.ceil(numel / float(self.block_size)) if numel > 0 else 0
            if self.nbits == 4:
                packed_bytes = math.ceil(numel / 2.0)
            elif self.nbits == 8:
                packed_bytes = numel
            else:
                packed_bytes = math.ceil(numel * self.nbits / 8.0)
            scale_bytes = num_blocks * 4  # float32 scales
            quant_bytes += packed_bytes + scale_bytes

        ratio = float(orig_bytes) / max(quant_bytes, 1)
        savings = (1.0 - (quant_bytes / max(orig_bytes, 1))) * 100.0

        return {
            "orig_bytes": orig_bytes,
            "quant_bytes": quant_bytes,
            "compression_ratio": ratio,
            "bandwidth_savings_pct": savings,
        }
