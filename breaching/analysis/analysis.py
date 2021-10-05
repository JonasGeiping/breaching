"""Simple report function based on PSNR and maybe SSIM and maybe better ideas..."""
import torch


from .metrics import psnr_compute, registered_psnr_compute, image_identifiability_precision


def report(reconstructed_user_data, true_user_data, server_payload, model, dataloader=None,
           setup=dict(device=torch.device('cpu'), dtype=torch.float), order_batch=True, compute_full_iip=False,
           skip_rpsnr=True):
    import lpips   # lazily import this only if report is used.
    lpips_scorer = lpips.LPIPS(net='alex').to(**setup)

    dm = torch.as_tensor(server_payload['data'].mean, **setup)[None, :, None, None]
    ds = torch.as_tensor(server_payload['data'].std, **setup)[None, :, None, None]
    model.to(**setup)

    rec_denormalized = torch.clamp(reconstructed_user_data['data'] * ds + dm, 0, 1)
    ground_truth_denormalized = torch.clamp(true_user_data['data'] * ds + dm, 0, 1)

    if order_batch:
        order = compute_batch_order(lpips_scorer, rec_denormalized, ground_truth_denormalized, setup)
        reconstructed_user_data['data'] = reconstructed_user_data['data'][order]
        if reconstructed_user_data['labels'] is not None:
            reconstructed_user_data['labels'] = reconstructed_user_data['labels'][order]
        rec_denormalized = rec_denormalized[order]
    else:
        order = None


    test_mse = (rec_denormalized - ground_truth_denormalized).pow(2).mean().item()
    test_psnr = psnr_compute(rec_denormalized, ground_truth_denormalized, factor=1).item()

    # Hint: This part switches to the lpips [-1, 1] normalization:
    test_lpips = lpips_scorer(rec_denormalized, ground_truth_denormalized, normalize=True).mean().item()

    # Compute registered psnr. This is a bit computationally intensive:
    if not skip_rpsnr:
        test_rpsnr = registered_psnr_compute(rec_denormalized.cpu(), ground_truth_denormalized.cpu(), factor=1).item()
    else:
        test_rpsnr = float('nan')

    # Compute IIP score if a dataloader is passed:
    if dataloader is not None:
        if compute_full_iip:
            scores = ['pixel', 'lpips', 'self']
        else:
            scores = ['pixel']
        iip_scores = image_identifiability_precision(reconstructed_user_data, true_user_data, dataloader,
                                                     lpips_scorer=lpips_scorer, model=model, scores=scores)
    else:
        iip_scores = dict(none=float('NaN'))

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

    # Record model parameters:
    parameters = sum([p.numel() for p in model.parameters()])

    # Print report:
    iip_scoring = ' | '.join([f'IIP-{k}: {v:5.2%}' for k, v in iip_scores.items()])
    print(f"METRICS: | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} | LPIPS: {test_lpips:4.2f}|"
          f" R-PSNR: {test_rpsnr:4.2f} | {iip_scoring}")

    metrics = dict(mse=test_mse, psnr=test_psnr, feat_mse=feat_mse, lpips=test_lpips, rpsnr=test_rpsnr,
                   order=order, **{f'IIP-{k}': v for k, v in iip_scores.items()}, parameters=parameters)
    return metrics


def compute_batch_order(lpips_scorer, rec_denormalized, ground_truth_denormalized, setup):
    """Re-order a batch of images according to LPIPS statistics of source batch, trying to match similar images.

    This implementation basically follows the LPIPS.forward method, but for an entire batch."""
    from scipy.optimize import linear_sum_assignment  # Again a lazy import


    B = rec_denormalized.shape[0]
    L = lpips_scorer.L
    assert ground_truth_denormalized.shape[0] == B

    with torch.inference_mode():
        # Compute all features [assume sufficient memory is a given]
        features_rec = []
        for input in rec_denormalized:
            input_scaled = lpips_scorer.scaling_layer(input)
            output = lpips_scorer.net.forward(input_scaled)
            layer_features = {}
            for kk in range(L):
                layer_features[kk] = normalize_tensor(output[kk])
            features_rec.append(layer_features)

        features_gt = []
        for input in ground_truth_denormalized:
            input_scaled = lpips_scorer.scaling_layer(input)
            output = lpips_scorer.net.forward(input_scaled)
            layer_features = {}
            for kk in range(L):
                layer_features[kk] = normalize_tensor(output[kk])
            features_gt.append(layer_features)

        # Compute overall similarities:
        similarity_matrix = torch.zeros(B, B, **setup)
        for idx, x in enumerate(features_gt):
            for idy, y in enumerate(features_rec):
                for kk in range(L):
                    diff = (x[kk] - y[kk])**2
                    similarity_matrix[idx, idy] += spatial_average(lpips_scorer.lins[kk](diff)).squeeze()

    _, rec_assignment = linear_sum_assignment(similarity_matrix.cpu().numpy(), maximize=False)
    return torch.as_tensor(rec_assignment, device=setup['device'], dtype=torch.long)


def normalize_tensor(in_feat, eps=1e-10):
    """From https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/__init__.py."""
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    """ https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py ."""
    return in_tens.mean([2, 3], keepdim=keepdim)
