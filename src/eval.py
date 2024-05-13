import datetime
import os
from pathlib import Path
from typing import List, Tuple, Optional

import hydra
import pyrootutils
from lightning import LightningDataModule, LightningModule, Trainer, Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.nn import Module
import lightning as L
import wandb
import torch
from omegaconf import OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src.lightning_modules.verification_embedding_module import VerificationEmbeddingModule
from src import utils
from src.datamodules.base_datamodule import BaseDatamodule
from src.setup_utils import setup_model, setup_datamodule, setup_lightning_module, download_checkpoint_with_run_config, \
    get_hydra_config_from_folder

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    wandb.init(project="deep-vr-user-verification", group="eval")

    api = wandb.Api(
        overrides={
            'entity': '99fe99',
            'project': 'deep-vr-user-verification'
        }
    )
    model_output_dir = Path(cfg.paths.root_dir).joinpath("model_outputs")
    run_id = download_checkpoint_with_run_config(
        api.run(f"99fe99/deep-vr-user-verification/{cfg.run_id}"),
        model_output_dir
    )
    if run_id is None:
        log.info(f"Could not download checkpoint")
        return {}, {}
    folder = model_output_dir.joinpath(run_id)
    model_hydra_config = get_hydra_config_from_folder(folder)
    log.info("Instantiating loggers...")
    # logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
    wandb_logger = hydra.utils.instantiate(model_hydra_config.logger.wandb)
    model_hydra_config.group = cfg.group
    model_hydra_config.original_run_id = run_id
    wandb.config.update(OmegaConf.to_container(model_hydra_config))

    if model_hydra_config.get("torch_matmul_precision"):
        torch.set_float32_matmul_precision(model_hydra_config.torch_matmul_precision)
    # set seed for random number generators in pytorch, numpy and python.random
    if model_hydra_config.get("seed"):
        L.seed_everything(model_hydra_config.seed, workers=True)

    log.info(f"Loading checkpoint from <{model_hydra_config.ckpt_path}>")
    model = VerificationEmbeddingModule.load_from_checkpoint(model_hydra_config.ckpt_path)
    model.verification_head.best_threshold = model_hydra_config.verification_head.threshold
    cfg.datamodule.data_stats_path = model_hydra_config.datamodule.data_stats_path
    datamodule: Optional[BaseDatamodule] = setup_datamodule(cfg.datamodule, "test", 60)

    log.info(f"Instantiating trainer <{model_hydra_config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(model_hydra_config.trainer, logger=wandb_logger)

    object_dict = {
        "cfg": model_hydra_config,
        "datamodule": datamodule,
        "model": model,
        "logger": wandb_logger,
        "trainer": trainer,
    }

    utils.log_hyperparameters(
        cfg=model_hydra_config,
        model=model,
        trainer=trainer,
    )

    log.info("Starting testing!")
    trainer.test(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path=model_hydra_config.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    try:
        print(cfg.output_dir)
    except Exception:
        folder = os.path.dirname(
            os.path.dirname(__file__)) + f"/logs/eval-{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')}"
        os.makedirs(folder, exist_ok=True)
        cfg.paths.output_dir = folder
        cfg.paths.work_dir = os.path.dirname(__file__)
    utils.extras(cfg)

    evaluate(cfg)


def call_main_externally():
    main()


if __name__ == "__main__":
    main()
