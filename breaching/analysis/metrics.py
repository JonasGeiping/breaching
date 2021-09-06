"""Various metrics."""
import torch


def psnr_compute(img_batch, ref_batch, batched=False, factor=1.0, clip=False):
    """Standard PSNR."""
    if clip:
        img_batch = torch.clamp(img_batch, 0, 1)

    if batched:
        mse = ((img_batch.detach() - ref_batch)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return torch.tensor(float('nan'), device=img_batch.device)
        else:
            return torch.tensor(float('inf'), device=img_batch.device)
    else:
        B = img_batch.shape[0]
        mse_per_example = ((img_batch.detach() - ref_batch)**2).view(B, -1).mean(dim=1)
        if any(mse_per_example == 0):
            return torch.tensor(float('inf'), device=img_batch.device)
        elif not all(torch.isfinite(mse_per_example)):
            return torch.tensor(float('nan'), device=img_batch.device)
        else:
            return (10 * torch.log10(factor**2 / mse_per_example)).mean()
