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


def registered_psnr_compute(img_batch, ref_batch, factor=1.0):
    """Use ORB features to register images onto reference before computing PSNR scores."""
    import skimage.feature  # Lazy metric stuff import
    import skimage.measure
    import skimage.transform


    descriptor_extractor = skimage.feature.ORB(n_keypoints=800)

    psnr_vals = torch.zeros(img_batch.shape[0])
    for idx, (img, ref) in enumerate(zip(img_batch, ref_batch)):
        default_psnr = psnr_compute(img, ref, factor=1.0, batched=True)
        try:
            img_np, ref_np = img.numpy(), ref.numpy()  # move to numpy
            descriptor_extractor.detect_and_extract(ref_np.mean(axis=0))  # and grayscale for ORB
            keypoints_src, descriptors_src = descriptor_extractor.keypoints, descriptor_extractor.descriptors
            descriptor_extractor.detect_and_extract(img_np.mean(axis=0))
            keypoints_tgt, descriptors_tgt = descriptor_extractor.keypoints, descriptor_extractor.descriptors

            matches = skimage.feature.match_descriptors(descriptors_src, descriptors_tgt, cross_check=True)
            # Look for an affine transform and search with RANSAC over matches:
            model_robust, inliers = skimage.measure.ransac((keypoints_tgt[matches[:, 1]],
                                                           keypoints_src[matches[:, 0]]), skimage.transform.EuclideanTransform,
                                                           min_samples=len(matches) - 1, residual_threshold=4, max_trials=2500)  # :>
            warped_img = skimage.transform.warp(img_np.transpose(1, 2, 0), model_robust, mode='wrap', order=1)
            # Compute normal PSNR from here:
            registered_psnr = psnr_compute(torch.as_tensor(warped_img), ref.permute(1, 2, 0), factor=1.0, batched=True)
            if registered_psnr.isfinite():
                psnr_vals[idx] = max(registered_psnr, default_psnr)
            else:
                psnr_vals[idx] = default_psnr
        except (TypeError, IndexError, RuntimeError):
            # TypeError if RANSAC fails
            # IndexError if not enough matches are found
            # RunTimeError if ORB does not find enough features
            psnr_vals[idx] = default_psnr
    return psnr_vals.mean()
