"""Simple report function based on PSNR and maybe SSIM and maybe better ideas..."""
import torch


from .metrics import psnr_compute, registered_psnr_compute


def report(reconstructed_user_data, true_user_data, server_payload, model, setup):
    import lpips   # lazily import this only if report is used.
    lpips_scorer = lpips.LPIPS(net='alex')

    dm = torch.as_tensor(server_payload['data'].mean, **setup)[None, :, None, None]
    ds = torch.as_tensor(server_payload['data'].std, **setup)[None, :, None, None]
    model.to(**setup)


    rec_denormalized = torch.clamp(reconstructed_user_data['data'] * ds + dm, 0, 1).cpu()
    ground_truth_denormalized = torch.clamp(true_user_data['data'] * ds + dm, 0, 1).cpu()

    test_mse = (rec_denormalized - ground_truth_denormalized).pow(2).mean().item()
    test_psnr = psnr_compute(rec_denormalized, ground_truth_denormalized, factor=1).item()

    # Hint: This part switches to the lpips [-1, 1] normalization:
    test_lpips = lpips_scorer(rec_denormalized, ground_truth_denormalized, normalize=True).mean().item()

    # Compute registered psnr. This is a bit computationally intensive:
    test_rpsnr = registered_psnr_compute(rec_denormalized, ground_truth_denormalized, factor=1).item()


    feat_mse = 0.0
    for payload in server_payload['queries']:
        parameters = payload['parameters']
        buffers = payload['buffers']

        with torch.no_grad():
            for param, server_state in zip(model.parameters(), parameters):
                param.copy_(server_state.to(**setup))
            for buffer, server_state in zip(model.buffers(), buffers):
                buffer.copy_(server_state.to(**setup))

            # Compute the forward passes
            feat_mse += (model(reconstructed_user_data['data']) - model(true_user_data['data'])).pow(2).mean().item()

    # Print report:
    print(f"METRICS: | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} | LPIPS: {test_lpips:4.2f}|"
          f" R-PSNR: {test_rpsnr:4.2f}")

    metrics = dict(mse=test_mse, psnr=test_psnr, feat_mse=feat_mse, lpips=test_lpips, rpsnr=test_rpsnr)
    return metrics
