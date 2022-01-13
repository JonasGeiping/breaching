"""This script computes a suite of benchmark numbers for the given attack.


The arguments from the default config carry over here.
"""

import hydra
from omegaconf import OmegaConf, open_dict

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

    # hardcoding for 
    with open_dict(cfg):
        cfg.case.server.name = 'class_malicious_parameters'
        cfg.case.impl.sharing_strategy = 'file_system'
        
        cfg.case.user.provide_labels = True
        cfg.case.user.provide_buffers = True
        cfg.case.user.provide_num_data_points = True

    log.info(OmegaConf.to_yaml(cfg))
    breaching.utils.initialize_multiprocess_log(cfg)  # manually save log configuration
    main_process(0, 1, cfg)

    log.info("-------------------------------------------------------------")
    log.info(
        f"Finished computations with total train time: " f"{str(datetime.timedelta(seconds=time.time() - launch_time))}"
    )
    log.info("-----------------Job finished.-------------------------------")


def main_process(process_idx, local_group_size, cfg, num_trials=1):
    """This function controls the central routine."""
    local_time = time.time()
    setup = breaching.utils.system_startup(process_idx, local_group_size, cfg)
    model, loss_fn = breaching.cases.construct_model(cfg.case.model, cfg.case.data, cfg.case.server.pretrained)

    server = breaching.cases.construct_server(model, loss_fn, cfg.case, setup)
    model = server.vet_model(model)
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg.attack, setup)
    if cfg.case.user.user_idx is not None:
        print("The argument user_idx is disregarded during the benchmark. Data selection is fixed.")

    if cfg.case.user.num_data_points == 1:
        cfg.case.data.partition == "unique-class"  # Different label per user
    else:
        cfg.case.data.partition == "balanced"  # Balanced partition of labels

    cfg.case.user.user_idx = 0
    overall_metrics = []
    for run in range(num_trials):
        # Select data that has not been seen before:
        cfg.case.user.user_idx += 1
        user = breaching.cases.construct_user(model, loss_fn, cfg.case, setup)
        log.info(f"Now evaluating indices {user.user_idx} in trial {run}.")

        user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
        attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)

        # get info first (not count as one attack?)
        server_payload = server.distribute_payload()
        shared_data, true_user_data = user.compute_local_updates(server_payload)
        t_labels = shared_data["metadata"]["labels"].detach().cpu().numpy()

        reconstruction_data = torch.zeros_like(true_user_data["data"])

        # attack cls by cls
        for target_cls in np.unique(t_labels):
            target_indx = np.where(t_labels == target_cls)
            tmp_shared_data = copy.deepcopy(shared_data)
            tmp_shared_data['metadata']['num_data_points'] = len(target_indx)
            tmp_shared_data['metadata']['labels'] = shared_data['metadata']['labels'][target_indx]

            if len(target_indx) == 1:
                # simple cls attack if there is no cls collision
                reconstruction_data_i, stats = simple_cls_attack(user, server, attacker, tmp_shared_data, cfg)
            else:
                raise NotImplementedError("Haven't implement cls collision now!")
            
            reconstruction_data_i = reconstruction_data_i["data"]
            reconstruction_data[target_indx] = reconstruction_data_i

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
        breaching.utils.save_summary(cfg, metrics, stats, time.time() - local_time, original_cwd=False)
        overall_metrics.append(metrics)
        if cfg.dryrun:
            break

    # Compute average statistics:
    average_metrics = breaching.utils.avg_n_dicts(overall_metrics)

    # Save global summary:
    breaching.utils.save_summary(
        cfg, average_metrics, stats, time.time() - local_time, original_cwd=True, table_name="CLASSATTACK_breach"
    )

def simple_cls_attack(user, server, attacker, shared_data, cfg):
    cls_to_obtain = int(shared_data['metadata']['labels'][0])
    extra_info = {'cls_to_obtain': cls_to_obtain}
    
    # modify the parameters first
    server.reset_model()
    server.reconfigure_model('cls_attack', extra_info=extra_info)

    server_payload = server.distribute_payload()
    tmp_shared_data, _ = user.compute_local_updates(server_payload)
    shared_data['gradients'] = tmp_shared_data['gradients']

    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], server.secrets, dryrun=cfg.dryrun)

    return reconstructed_user_data, stats

if __name__ == "__main__":
    main_launcher()
