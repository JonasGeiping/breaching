"""This script computes a suite of benchmark numbers for the given attack.


The arguments from the default config carry over here.
"""

import hydra
from omegaconf import OmegaConf

import datetime
import time
import logging

import breaching

import os

import numpy as np
import torch
import copy

os.environ["HYDRA_FULL_ERROR"] = "0"
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="cfg")
def main_launcher(cfg):
    """This is boiler-plate code for the launcher."""

    log.info("--------------------------------------------------------------")
    log.info("-----Launching federating learning breach experiment! --------")

    launch_time = time.time()
    if cfg.seed is None:
        cfg.seed = 233  # The benchmark seed is fixed by default!

    log.info(OmegaConf.to_yaml(cfg))
    breaching.utils.initialize_multiprocess_log(cfg)  # manually save log configuration
    main_process(0, 1, cfg)

    log.info("-------------------------------------------------------------")
    log.info(
        f"Finished computations with total train time: " f"{str(datetime.timedelta(seconds=time.time() - launch_time))}"
    )
    log.info("-----------------Job finished.-------------------------------")


def main_process(process_idx, local_group_size, cfg, num_trials=100, target_max_psnr=True, opt_on_avg_grad=False):
    """This function controls the central routine."""
    local_time = time.time()
    setup = breaching.utils.system_startup(process_idx, local_group_size, cfg)

    if cfg.num_trials is not None:
        num_trials = cfg.num_trials

    if "opt_on_avg_grad" in cfg:
        opt_on_avg_grad = cfg.opt_on_avg_grad

    model, loss_fn = breaching.cases.construct_model(cfg.case.model, cfg.case.data, cfg.case.server.pretrained)

    server = breaching.cases.construct_server(model, loss_fn, cfg.case, setup)
    model = server.vet_model(model)
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg.attack, setup)
    if cfg.case.user.user_idx is not None:
        print("The argument user_idx is disregarded during the benchmark. Data selection is fixed.")
    log.info(
        f"Partitioning is set to {cfg.case.data.partition}. Make sure there exist {num_trials} users in this scheme."
    )

    cfg.case.user.user_idx = -1
    overall_metrics = []
    for run in range(num_trials):
        # Select data that has not been seen before:
        cfg.case.user.user_idx += 1
        user = breaching.cases.construct_user(model, loss_fn, cfg.case, setup)
        log.info(f"Now evaluating user with idx {user.user_idx} in trial {run}.")

        attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)

        # get class info first (not count as one query?)
        server_payload = server.distribute_payload()
        shared_data, true_user_data = user.compute_local_updates(server_payload)
        t_labels = shared_data["metadata"]["labels"].detach().cpu().numpy()
        log.info(f"Found labels {t_labels} in first query.")

        if opt_on_avg_grad:
            # optimize on averaged gradient with cls attack
            log.info("Optimize on averaged gradient with cls attack.")

            # cls attack on all labels in the batch
            extra_info = {"cls_to_obtain": t_labels}
            server.reconfigure_model("cls_attack", extra_info=extra_info)
            server_payload = server.distribute_payload()
            shared_data, true_user_data = user.compute_local_updates(server_payload)

            reconstruction, stats = attacker.reconstruct(
                [server_payload], [shared_data], server.secrets, dryrun=cfg.dryrun
            )
        else:
            # attack cls by cls
            log.info("Attack cls by cls cls attack.")
            reconstruction_data = torch.zeros_like(true_user_data["data"])
            for target_cls in np.unique(t_labels):
                target_indx = np.where(t_labels == target_cls)[0]
                tmp_shared_data = copy.deepcopy(shared_data)
                tmp_shared_data["metadata"]["num_data_points"] = len(target_indx)
                tmp_shared_data["metadata"]["labels"] = shared_data["metadata"]["labels"][target_indx]

                if len(target_indx) == 1:
                    # simple cls attack if there is no cls collision
                    log.info(f"Attacking label {tmp_shared_data['metadata']['labels'].item()} with cls attack.")
                    reconstruction_data_i, stats = simple_cls_attack(user, server, attacker, tmp_shared_data, cfg)
                else:
                    # send several queries because of cls collision
                    log.info(f"Attacking label {tmp_shared_data['metadata']['labels'][0].item()} with binary attack.")
                    reconstruction_data_i, stats = cls_collision_attack(
                        user,
                        server,
                        attacker,
                        tmp_shared_data,
                        cfg,
                        target_max_psnr,
                        copy.deepcopy(reconstruction_data[target_indx]),
                    )

                reconstruction_data_i = reconstruction_data_i["data"]
                reconstruction_data[target_indx] = reconstruction_data_i

                if target_max_psnr:
                    break

            reconstruction = {"data": reconstruction_data, "labels": shared_data["metadata"]["labels"]}

        # Run the full set of metrics:
        metrics = breaching.analysis.report(
            reconstruction,
            true_user_data,
            [server_payload],
            server.model,
            order_batch=True,
            compute_full_iip=True,
            compute_rpsnr=True,
            compute_ssim=True,
            cfg_case=cfg.case,
            setup=setup,
        )

        # Save local summary:
        breaching.utils.save_summary(
            cfg, metrics, stats, user.counted_queries, time.time() - local_time, original_cwd=False
        )
        overall_metrics.append(metrics)
        # Save recovered data:
        if cfg.save_reconstruction:
            if target_max_psnr:
                sorted_indx = (metrics["order"].cpu() == torch.as_tensor(target_indx)).nonzero().squeeze()
            else:
                sorted_indx = None
            breaching.utils.save_reconstruction(
                reconstruction, [server_payload], true_user_data, cfg, target_indx=sorted_indx
            )
        if cfg.dryrun:
            break

    # Compute average statistics:
    average_metrics = breaching.utils.avg_n_dicts(overall_metrics)

    # Save global summary:
    breaching.utils.save_summary(
        cfg, average_metrics, stats, time.time() - local_time, None, original_cwd=True, table_name="CLASSATTACK_breach"
    )


def simple_cls_attack(user, server, attacker, shared_data, cfg):
    cls_to_obtain = int(shared_data["metadata"]["labels"][0])
    extra_info = {"cls_to_obtain": cls_to_obtain}

    # modify the parameters first
    server.reset_model()
    server.reconfigure_model("cls_attack", extra_info=extra_info)

    server_payload = server.distribute_payload()
    tmp_shared_data, _ = user.compute_local_updates(server_payload)
    shared_data["gradients"] = tmp_shared_data["gradients"]

    reconstructed_user_data, stats = attacker.reconstruct(
        [server_payload], [shared_data], server.secrets, dryrun=cfg.dryrun
    )

    return reconstructed_user_data, stats


def cls_collision_attack(user, server, attacker, shared_data, cfg, target_max_psnr, reconstruction_data):
    log.info(
        f"There are total {len(shared_data['metadata']['labels'])} datapoints with label {shared_data['metadata']['labels'][0].item()}."
    )

    cls_to_obtain = int(shared_data["metadata"]["labels"][0])
    extra_info = {"cls_to_obtain": cls_to_obtain}

    # find the starting point and the feature entry gives the max avg value
    server.reset_model()
    server.reconfigure_model("cls_attack", extra_info=extra_info)
    server_payload = server.distribute_payload()
    tmp_shared_data, _ = user.compute_local_updates(server_payload)
    avg_feature = torch.flatten(server.reconstruct_feature(tmp_shared_data, cls_to_obtain))
    
    while 6 > 5:
        feat_to_obtain = int(torch.argmax(avg_feature))
        feat_value = float(avg_feature[feat_to_obtain])

        # binary attack to recover all single gradients
        extra_info["feat_to_obtain"] = feat_to_obtain
        extra_info["feat_value"] = feat_value
        extra_info["multiplier"] = 1

        try:
            recovered_single_gradients = server.binary_attack(user, extra_info)
            break
        except Exception:
            avg_feature[feat_to_obtain] = -1000
            pass

    # return to the model with multiplier=1
    server.reset_model()
    extra_info["multiplier"] = 1
    extra_info["feat_value"] = feat_value
    server.reconfigure_model("cls_attack", extra_info=extra_info)
    server.reconfigure_model("feature_attack", extra_info=extra_info)
    server_payload = server.distribute_payload()

    # recover image by image
    # add reversed() because the ith is always more confident than i-1th
    for i, grad_i in enumerate(reversed(recovered_single_gradients)):
        log.info(f"Start recovering datapoint {i} of label {shared_data['metadata']['labels'][0].item()}.")

        tmp_share_data = copy.deepcopy(shared_data)
        tmp_share_data["metadata"]["num_data_points"] = 1
        tmp_share_data["metadata"]["labels"] = shared_data["metadata"]["labels"][0:1]
        tmp_share_data["gradients"] = grad_i

        reconstructed_user_data, stats = attacker.reconstruct(
            [server_payload], [tmp_share_data], server.secrets, dryrun=cfg.dryrun
        )

        reconstruction_data[[i]] = reconstructed_user_data["data"]

        if target_max_psnr:
            break

    return dict(data=reconstruction_data, labels=shared_data["metadata"]["labels"]), stats


if __name__ == "__main__":
    main_launcher()
