"""
Control script to launch some attack that attempts to breach privacy federated learning in a given case
with the overall goal of recovering user data, e.g. image data, directly as much as possible.
"""

import torch
import hydra
from omegaconf import OmegaConf

import datetime
import time
import logging

import breaching

import os
os.environ["HYDRA_FULL_ERROR"] = "0"
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="cfg")
def main_launcher(cfg):
    """This is boiler-plate code for the launcher."""

    log.info('--------------------------------------------------------------')
    log.info('-----Launching federating learning breach experiment! --------')

    launch_time = time.time()
    if cfg.seed is None:
        cfg.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    log.info(OmegaConf.to_yaml(cfg))
    breaching.utils.initialize_multiprocess_log(cfg)  # manually save log configuration
    main_process(0, 1, cfg)

    log.info('-------------------------------------------------------------')
    log.info(f'Finished computations with total train time: '
             f'{str(datetime.timedelta(seconds=time.time() - launch_time))}')
    log.info('-----------------Job finished.---------------------------d----')


def main_process(process_idx, local_group_size, cfg):
    """This function controls the central routine."""
    local_time = time.time()
    setup = breaching.utils.system_startup(process_idx, local_group_size, cfg)

    # Instantiate all parties
    user, server = breaching.cases.construct_case(cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)

    # Simulate an attacked FL protocol
    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)  # True user data is returned only for analysis
    reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, server.secrets, dryrun=cfg.dryrun)

    # How good is the reconstruction?
    metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, server_payload, server.model, setup)
    breaching.utils.save_summary(cfg, metrics, stats, time.time() - local_time)

    # breach.utils.save_image()

if __name__ == "__main__":
    main_launcher()
