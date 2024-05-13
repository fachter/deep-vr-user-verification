import pathlib

from omegaconf import OmegaConf
import wandb

from kube_configs.training_job_spawner import KubernetesHelper


def main():
    names = pathlib.Path("../wandb_sweep_configs/verification_head_search/threshold_head/").glob("kl_divergence_contrastive_loss.sweep.yaml")
    to_start = []
    project_name = "deep-vr-user-verification"
    for file_name in names:
        sweep_config = OmegaConf.load(file_name)
        job_name = sweep_config.name.replace("_", "-")
        sweep_id = wandb.sweep(OmegaConf.to_container(sweep_config), entity="99fe99", project=project_name)
        to_start.append((f"99fe99/{project_name}/{sweep_id}", job_name))

    for sweep_id, job_name in to_start:
        kube_helper = KubernetesHelper()
        kube_helper.create_jobs(2, "sweep_config.jinja2.yaml", {
            'job_name': f'train-{job_name}',
            'sweep_id': sweep_id,
            'gpus': 1,
            'cpus_request': 2,
            'memory_request': 100,
            'cpus_limit': 15,
            'memory_limit': 200,
            'num_workers': 8
        })


if __name__ == '__main__':
    main()
