import os
import pathlib
import re

import dotenv

from training_job_spawner import KubernetesHelper, script_args_to_string
import logging
import paramiko
from getpass import getpass


def start_single_run():
    kube_helper = KubernetesHelper()
    experiment_path = "verification/different_verification_heads/gru_distance_match_kl_div_soft_contrastive_pair_loss"
    experiment = "gru-kl-div-soft-contrastive-pair-loss"

    kube_helper.create_jobs(1, "single_run.jinja2.yaml", {
        'job_name': f'train-{experiment}',
        'experiment': f'{experiment_path}',
        'gpus': 1,
        'cpus_request': 5,
        'memory_request': 50,
        'cpus_limit': 10,
        'memory_limit': 100,
    })


def start_sweep():
    sweep_names = [
        "threshold-seed-search-data-shuffle",
        # "match-prob-soft-contrastive-kl-div",
        # "match-prob-soft-contrastive-lp-dist",
    ]
    sweep_ids = [
        "99fe99/deep-vr-user-verification/qoeaefkn",
        # "99fe99/deep-vr-user-verification/f5xp2c00",
        # "99fe99/deep-vr-user-verification/e38f8ukw",
    ]
    for sweep_id, sweep_name in zip(sweep_ids, sweep_names):
        # sweep_name = "threshold-ms-loss"
        # sweep_id = "99fe99/deep-vr-user-verification/grmnhm6o"

        kube_helper = KubernetesHelper()
        kube_helper.create_jobs(1, "sweep_config.jinja2.yaml", {
            'job_name': f'train-{sweep_name}',
            'sweep_id': sweep_id,
            'gpus': 1,
            'cpus_request': 2,
            'memory_request': 100,
            'cpus_limit': 15,
            'memory_limit': 200,
            'num_workers': 8
        })


def run_script():
    job_name = "train-kl-div-contrastive-97348-12345"
    script_name = f"scripts/schedule_threshold_head_search.sh"
    kube_helper = KubernetesHelper()
    kube_helper.create_jobs(1, "run_script_config.jinja2.yaml", {
        'job_name': f'{job_name}',
        'script_name': script_name,
        'gpus': 1,
        'cpus_request': 2,
        'memory_request': 50,
        'cpus_limit': 15,
        'memory_limit': 150,
    })


def run_multiple_scripts():
    scripts = list(pathlib.Path("../scripts/threshold_final_tests").glob("*.sh"))

    print("Found {} scripts".format(len(scripts)))
    for script in scripts:
        job_name = script.stem.replace("_", "-")
        script_name = f"scripts/threshold_final_tests/{script.name}"
        kube_helper = KubernetesHelper()
        kube_helper.create_jobs(1, "run_script_config.jinja2.yaml", {
            'job_name': f'test-sim-head-{job_name}',
            'script_name': script_name,
            'gpus': 1,
            'cpus_request': 2,
            'memory_request': 50,
            'cpus_limit': 15,
            'memory_limit': 150,
        })


def delete_all_jobs():
    kube_helper = KubernetesHelper()
    kube_helper.delete_jobs("test-sim-head-run-12345-12345", ignore_actives=False)


def save_all_logs():
    kube_helper = KubernetesHelper()
    kube_helper.save_logs("-")


if __name__ == '__main__':
    # start_single_run()
    # start_sweep()
    # save_all_logs()
    # delete_all_jobs()
    # run_script()
    run_multiple_scripts()
    pass
