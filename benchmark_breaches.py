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


def main_process(process_idx, local_group_size, cfg, num_trials=100):
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
        # Run exchange
        shared_user_data, payloads, true_user_data = server.run_protocol(user)
        # Evaluate attack:
        reconstruction, stats = attacker.reconstruct(payloads, shared_user_data, server.secrets, dryrun=cfg.dryrun)

        # Run the full set of metrics:
        metrics = breaching.analysis.report(
            reconstruction,
            true_user_data,
            payloads,
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
        cfg, average_metrics, stats, time.time() - local_time, original_cwd=True, table_name="BENCHMARK_breach"
    )


if __name__ == "__main__":
    main_launcher()
