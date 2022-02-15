"""Utility functions for class/feature fishing attacks."""

import numbers
import torch

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

import torchvision.transforms as transforms


default_setup = dict(device=torch.device("cpu"), dtype=torch.float)


def reconfigure_class_parameter_attack(model, original_model, model_state, extra_info={}):
    if model_state == "cls_attack" and "cls_to_obtain" in extra_info:
        cls_to_obtain = extra_info["cls_to_obtain"]
        cls_to_obtain = wrap_indices(cls_to_obtain)

        with torch.no_grad():
            *_, l_w, l_b = model.parameters()

            # linear weight
            masked_param = torch.zeros_like(l_w, dtype=l_w.dtype).to(l_w.device)
            masked_param[cls_to_obtain] = torch.ones_like(l_w[cls_to_obtain], dtype=l_w.dtype).to(l_w.device) * 0.5
            l_w.copy_(masked_param.to(l_w.device))

            # linear bias
            masked_param = torch.ones_like(l_b, dtype=l_b.dtype).to(l_b.device) * 1000
            masked_param[cls_to_obtain] = l_b[cls_to_obtain]
            l_b.copy_(masked_param.to(l_b.device))

    if model_state == "fishing_attack" and "cls_to_obtain" in extra_info:
        cls_to_obtain = extra_info["cls_to_obtain"]
        b_mv = extra_info["b_mv"] if "b_mv" in extra_info else 0
        b_mv_non = extra_info["b_mv_non"] if "b_mv_non" in extra_info else 0
        multiplier = extra_info["multiplier"] if "multiplier" in extra_info else 1
        cls_to_obtain = wrap_indices(cls_to_obtain)

        with torch.no_grad():
            *_, l_w, l_b = model.parameters()

            # linear weight
            masked_param = torch.zeros_like(l_w, dtype=l_w.dtype).to(l_w.device)
            masked_param[cls_to_obtain] = l_w[cls_to_obtain]
            l_w.copy_(masked_param.to(l_w.device))

            # linear bias
            masked_param = torch.ones_like(l_b, dtype=l_b.dtype).to(l_b.device)
            masked_param[cls_to_obtain] = l_b[cls_to_obtain] + b_mv
            l_b.copy_(masked_param.to(l_b.device))

            *_, l_w, l_b = model.parameters()
            *_, l_w_o, l_b_o = original_model.parameters()
            cls_to_obtain = int(extra_info["cls_to_obtain"])
            l_w[:cls_to_obtain] = l_w_o[:cls_to_obtain]
            l_w[cls_to_obtain + 1 :] = l_w_o[cls_to_obtain + 1 :]
            l_b[:cls_to_obtain] = l_b_o[:cls_to_obtain] + b_mv_non
            l_b[cls_to_obtain + 1 :] = l_b_o[cls_to_obtain + 1 :] + b_mv_non

            l_w *= multiplier
            l_b *= multiplier

    if model_state == "feature_attack" and "cls_to_obtain" in extra_info and "feat_to_obtain" in extra_info:
        cls_to_obtain = extra_info["cls_to_obtain"]
        feat_to_obtain = extra_info["feat_to_obtain"]
        cls_to_obtain = wrap_indices(cls_to_obtain)
        feat_to_obtain = wrap_indices(feat_to_obtain)

        with torch.no_grad():
            *_, bn_w, bn_b, l_w, l_b = model.parameters()

            if "feat_value" not in extra_info:
                # just turn off other features
                masked_param = torch.zeros_like(l_w, dtype=l_w.dtype).to(l_w.device)
                masked_param[cls_to_obtain, feat_to_obtain] = torch.ones_like(
                    l_w[cls_to_obtain, feat_to_obtain], dtype=l_w.dtype
                )
                l_w.copy_(masked_param.to(l_w.device))
            else:
                # do gradient amplification
                multiplier = extra_info["multiplier"]
                extra_b = extra_info["extra_b"] if "extra_b" in extra_info else 0
                non_target_logit = extra_info["non_target_logit"] if "non_target_logit" in extra_info else 0
                db_flip = extra_info["db_flip"] if "db_flip" in extra_info else 1

                masked_param = torch.zeros_like(l_w, dtype=l_w.dtype).to(l_w.device)
                masked_param[cls_to_obtain, feat_to_obtain] = (
                    torch.ones_like(l_w[cls_to_obtain, feat_to_obtain], dtype=l_w.dtype).to(l_w.device)
                    * multiplier
                    * db_flip
                )
                l_w.copy_(masked_param.to(l_w.device))

                masked_param = torch.zeros_like(l_b, dtype=l_b.dtype).to(l_b.device) + non_target_logit
                masked_param[cls_to_obtain] = (
                    torch.zeros_like(l_b[cls_to_obtain], dtype=l_b.dtype).to(l_b.device)
                    - extra_info["feat_value"] * multiplier * db_flip
                    + extra_b
                )
                l_b.copy_(masked_param.to(l_b.device))

    if model_state == "db_attack" and "cls_to_obtain" in extra_info:
        cls_to_obtain = extra_info["cls_to_obtain"]
        cls_to_obtain = wrap_indices(cls_to_obtain)
        db_multiplier = extra_info["db_multiplier"]
        multiplier = extra_info["multiplier"]
        db_flip = extra_info["db_flip"]

        with torch.no_grad():
            *_, bn_w, bn_b, l_w, l_b = model.parameters()

            # batch norm weight
            masked_param = bn_w
            bn_w.copy_(masked_param.to(bn_w.device))

            # batch norm bias
            masked_param = bn_b + l_w[cls_to_obtain[0]] * db_multiplier
            bn_b.copy_(masked_param.to(bn_b.device))

            # linear weight
            masked_param = torch.zeros_like(l_w, dtype=l_w.dtype).to(l_w.device)
            masked_param[cls_to_obtain] = l_w[cls_to_obtain] * multiplier * db_flip
            l_w.copy_(masked_param.to(l_w.device))

            # linear bias
            masked_param = torch.zeros_like(l_b, dtype=l_b.dtype).to(l_b.device)
            masked_param[cls_to_obtain] = l_b[cls_to_obtain] * db_flip
            l_b.copy_(masked_param.to(l_b.device))


def wrap_indices(indices):
    if isinstance(indices, numbers.Number):
        return [indices]
    else:
        return list(indices)


def check_with_tolerance(value, list, threshold=0.05):
    for i in list:
        if abs(value - i) < threshold:
            return True

    return False


def order_gradients(self, recovered_single_gradients, gt_single_gradients, setup=default_setup):
    single_gradients = []
    num_data = len(gt_single_gradients)

    for grad_i in recovered_single_gradients:
        single_gradients.append(torch.cat([torch.flatten(i) for i in grad_i]))

    similarity_matrix = torch.zeros(num_data, num_data, **setup)
    for idx, x in enumerate(single_gradients):
        for idy, y in enumerate(gt_single_gradients):
            similarity_matrix[idy, idx] = torch.nn.CosineSimilarity(dim=0)(y, x).detach()

    try:
        _, rec_assignment = linear_sum_assignment(similarity_matrix.cpu().numpy(), maximize=True)
    except ValueError:
        log.info(f"ValueError from similarity matrix {similarity_matrix.cpu().numpy()}")
        log.info("Returning trivial order...")
        rec_assignment = list(range(num_data))

    return [recovered_single_gradients[i] for i in rec_assignment]


def reconstruct_feature(shared_data, cls_to_obtain):
    if type(shared_data) is not list:
        shared_grad = shared_data["gradients"]
    else:
        shared_grad = shared_data

    weights = shared_grad[-2]
    bias = shared_grad[-1]
    grads_fc_debiased = weights / bias[:, None]

    if bias[cls_to_obtain] != 0:
        return grads_fc_debiased[cls_to_obtain]
    else:
        return torch.zeros_like(grads_fc_debiased[0])


def cal_single_gradients(model, loss_fn, true_user_data, setup=default_setup):
    true_data = true_user_data["data"]
    num_data = len(true_data)
    labels = true_user_data["labels"]
    model = model.to(**setup)

    single_gradients = []
    single_losses = []

    for ii in range(num_data):
        cand_ii = true_data[ii : (ii + 1)]
        label_ii = labels[ii : (ii + 1)]
        model.zero_grad()
        spoofed_loss_ii = loss_fn(model(cand_ii), label_ii)
        gradient_ii = torch.autograd.grad(spoofed_loss_ii, model.parameters())
        gradient_ii = [g_ii.reshape(-1) for g_ii in gradient_ii]
        gradient_ii = torch.cat(gradient_ii)
        single_gradients.append(gradient_ii)
        single_losses.append(spoofed_loss_ii)

    return single_gradients, single_losses


def print_gradients_norm(single_gradients, single_losses, which_to_recover=-1, return_results=False):
    grad_norm = []
    losses = []

    if not return_results:
        print("grad norm   |   loss")

    for i, gradient_ii in enumerate(single_gradients):
        if not return_results:
            if i == which_to_recover:
                print(f"{float(torch.norm(gradient_ii)):2.4f} | {float(single_losses[i]):4.2f} - target")
            else:
                print(f"{float(torch.norm(gradient_ii)):2.4f} | {float(single_losses[i]):4.2f}")

        grad_norm.append(float(torch.norm(gradient_ii)))
        losses.append(float(single_losses[i]))

    if return_results:
        return torch.stack(grad_norm), torch.stack(losses)


def random_transformation(img):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img.shape[-2:], scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=1),
            # transforms.RandomVerticalFlip(p=1),
            transforms.GaussianBlur(3),
        ]
    )

    return transform(img)


def estimate_gt_stats(est_features, sample_sizes, indx=0):
    aggreg_data = []
    est_feature = est_features[indx]

    for i in range(len(est_feature)):
        feat_i = est_feature[i]
        size_i = sample_sizes[i]
        aggreg_data.append(feat_i * (size_i ** (1 / 2)))

    return np.mean(est_feature), np.std(aggreg_data)


def find_best_feat(est_features, sample_sizes, method="kstest"):
    if "kstest" in method:
        statistics = []
        for i in range(len(est_features)):
            tmp_series = est_features[i]
            tmp_series = (tmp_series - np.mean(tmp_series)) / np.std(tmp_series)
            statistics.append(stats.kstest(tmp_series, "norm")[0])

        return np.argmin(statistics)
    elif "most-spread" in method or "most-high-mean" in method:
        means = []
        stds = []
        for i in range(len(est_features)):
            mu, sigma = estimate_gt_stats(est_features, sample_sizes, indx=1)
            means.append(mu)
            stds.append(sigma)

        if "most-spread" in method:
            return np.argmax(stds)
        else:
            return np.argmax(means)
    else:
        raise ValueError(f"Method {method} not implemented.")

    return np.argmax(p_values)
