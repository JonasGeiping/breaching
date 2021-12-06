"""This script computes a suite of benchmark numbers for the given attack.


The arguments from the default config carry over here.
"""

import hydra
from omegaconf import OmegaConf

import datetime
import time
import logging

import breaching
import random

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
        cfg.seed = 233  # The benchmark seed is fixed by default!

    log.info(OmegaConf.to_yaml(cfg))
    breaching.utils.initialize_multiprocess_log(cfg)  # manually save log configuration
    main_process(0, 1, cfg)

    log.info('-------------------------------------------------------------')
    log.info(f'Finished computations with total train time: '
             f'{str(datetime.timedelta(seconds=time.time() - launch_time))}')
    log.info('-----------------Job finished.-------------------------------')


def main_process(process_idx, local_group_size, cfg, num_trials=2):
    """This function controls the central routine."""
    local_time = time.time()
    setup = breaching.utils.system_startup(process_idx, local_group_size, cfg)

    user, server = breaching.cases.construct_case(cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    if user.data_idx is not None:
        print('The argument data_idx is disregarded during the benchmark. Data selection is fixed.')

    batch_size = user.num_data_points
    if batch_size == 1:
        # Simply iterate through successive image classes without randomness. [Similar to the setup in InvertingGradients]
        offset = len(user.dataloader.dataset) // len(user.dataloader.dataset.classes)
        indices = [[idx * 50] for idx in range(num_trials)]
    else:
        # Benchmark: The first 397 classes are all animals, which makes this reasonably safe to do on ImageNet.
        # safe_subset = torch.arange(0, 397 * 50)
        subset = range(len(user.dataloader.dataset))
        raw_indices = random.sample(subset, batch_size * num_trials)
        indices = [raw_indices[idx * batch_size : (idx + 1) * batch_size] for idx in range(num_trials)]

    overall_metrics = []
    for run in range(num_trials):
        # Select data that has not been seen before:
        user.data_idx = indices[run]
        log.info(f'Now evaluating indices {user.data_idx} in trial {run}.')
        # Run exchange
        server_payload = server.distribute_payload()
        shared_data, true_user_data = user.compute_local_updates(server_payload)
        # Evaluate attack:
        reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, server.secrets, dryrun=cfg.dryrun)

        # Run the full set of metrics:
        metrics = breaching.analysis.report(reconstructed_user_data, true_user_data,
                                            server_payload, server.model, user.dataloader, setup,
                                            order_batch=True, compute_full_iip=True,
                                            compute_rpsnr=True, compute_ssim=True)

        # Save local summary:
        breaching.utils.save_summary(cfg, metrics, stats, time.time() - local_time, original_cwd=False)
        overall_metrics.append(metrics)

    # Compute average statistics:
    average_metrics = breaching.utils.avg_n_dicts(overall_metrics)

    # Save global summary:
    breaching.utils.save_summary(cfg, average_metrics, stats, time.time() - local_time,
                                 original_cwd=True, table_name='BENCHMARK_breach')

if __name__ == "__main__":
    main_launcher()
