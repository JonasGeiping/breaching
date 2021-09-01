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

import attacks
import cases

import os
os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="cfg")
def main_launcher(cfg):

    log.info('---------------------------------------------------')
    log.info('-----Launching federating learning breach experiment! --------')

    launch_time = time.time()
    if cfg.seed is None:
        cfg.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    ngpus_per_node = torch.cuda.device_count()
    log.info(OmegaConf.to_yaml(cfg))
    cases.utils.initialize_multiprocess_log(cfg)  # manually save log configuration
    main_process(0, 1, cfg)

    log.info('---------------------------------------------------')
    log.info(f'Finished computations with total train time: '
             f'{str(datetime.timedelta(seconds=time.time() - launch_time))}')
    log.info('-----------------Job finished.---------------------')


def main_process(process_idx, local_group_size, cfg):
    local_time = time.time()
    setup = cases.utils.system_startup(process_idx, local_group_size, cfg)

    case = cases.construct_case(cfg.case, setup)
    attacker = attacks.prepare_attack(case.model, cfg.attack, setup)

    stats = attacker.attack(case)

    if cases.utils.is_main_process():
        cases.utils.save_summary(cfg, stats, time.time() - local_time)


if __name__ == "__main__":
    main_launcher()
