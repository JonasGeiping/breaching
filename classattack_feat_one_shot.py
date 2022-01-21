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
from scipy import stats

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

    cfg.case.data.name = "ImageNet"
    cfg.case.data.examples_from_split = "train"
    cfg.case.data.default_clients = 1000

    log.info(OmegaConf.to_yaml(cfg))
    breaching.utils.initialize_multiprocess_log(cfg)  # manually save log configuration
    num_of_catches, num_trials = main_process(0, 1, cfg)

    log.info("-----------------Final Results-------------------------------")
    log.info(f"Total {num_trials} attempts")
    log.info(f"P(catch 1) = {round(len(num_of_catches[num_of_catches == 1]) / num_trials, 2)}")
    log.info(f"P(catch 2) = {round(len(num_of_catches[num_of_catches == 2]) / num_trials, 2)}")
    log.info(f"P(catch 3) = {round(len(num_of_catches[num_of_catches == 3]) / num_trials, 2)}")
    log.info(f"P(catch >= 4) = {round(len(num_of_catches[num_of_catches >= 4]) / num_trials, 2)}")

    log.info("-------------------------------------------------------------")
    log.info(
        f"Finished computations with total train time: " f"{str(datetime.timedelta(seconds=time.time() - launch_time))}"
    )
    log.info("-----------------Job finished.-------------------------------")


def main_process(
    process_idx, local_group_size, cfg, num_trials=100, num_to_est=900, batch_size=4):
    """This function controls the central routine."""
    local_time = time.time()
    setup = breaching.utils.system_startup(process_idx, local_group_size, cfg)

    if cfg.num_trials is not None:
        num_trials = cfg.num_trials

    if "num_to_est" in cfg:
        num_to_est = cfg.num_to_est

    if "batch_size" in cfg:
        batch_size = cfg.batch_size

    model, loss_fn = breaching.cases.construct_model(cfg.case.model, cfg.case.data, cfg.case.server.pretrained)

    server = breaching.cases.construct_server(model, loss_fn, cfg.case, setup)
    model = server.vet_model(model)
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg.attack, setup)
    attacker.objective.cfg_impl = cfg.attack.impl

    counter = 0
    num_of_catches = []
    for target_label in range(1000):
        cfg.case.data.target_label = target_label
        cls_to_obtain = cfg.case.data.target_label
        cfg.case.user.num_data_points = batch_size
        cfg.case.data.num_data_points = cfg.case.user.num_data_points
        extra_info = {'cls_to_obtain': cls_to_obtain}
        server.reset_model()
        server.reconfigure_model('cls_attack', extra_info=extra_info)
        
        log.info(f"Now estimating features from cls {cls_to_obtain} within {num_to_est} datapoints in batch size {batch_size}.")

        est_features, est_sample_sizes = server.estimate_feat(cfg, extra_info, num_to_est=num_to_est)
        f_indx = server.find_best_feat(est_features, est_sample_sizes, method="kstest")
        est_mean, est_std = server.estimate_gt_stats(est_features, est_sample_sizes, indx=f_indx)

        # modify the model
        server.reset_model()
        extra_info['multiplier'] = 300
        extra_info["feat_to_obtain"] = f_indx
        extra_info['feat_value'] = stats.norm.ppf(0.1, est_mean, est_std)
        server.reconfigure_model('cls_attack', extra_info=extra_info)
        server.reconfigure_model('feature_attack', extra_info=extra_info)

        log.info(f"Now start one-shot attack")

        num_to_test = 1000 - num_to_est
        start_ind = num_to_est // 10

        for i in range((num_to_test // 10) - 1):
            cfg.case.user.user_idx = start_ind + i

            log.info(f"----------- Now attack uer {i} ---------------")
            
            cfg.case.data.num_data_points = cfg.case.user.num_data_points
            user = breaching.cases.construct_user(model, loss_fn, cfg.case, setup)
            server_payload = server.distribute_payload()
            shared_data, true_user_data = user.compute_local_updates(server_payload)
            
            logtis = server.model(true_user_data["data"])[:, [cls_to_obtain]]
            num_of_catch = len(logtis[logtis <= 0])
            num_of_catches.append(num_of_catch)
            log.info(f"catches {num_of_catch} imgs!")

            counter += 1
            if counter == num_trials:
                return num_of_catches, num_trials

if __name__ == "__main__":
    main_launcher()
