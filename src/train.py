import pathlib
import signal
from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
import wandb
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module

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

from src import utils
from src.callbacks.log_best_metrics_callback import LogBestMetrics
from src.callbacks.user_halt_callback import UserHaltCallback
from src.datamodules.base_datamodule import BaseDatamodule
from src.verification_heads import VerificationHeadBase
from src.setup_utils import setup_datamodule, setup_model, setup_lightning_module, initialize_callbacks

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    wandb.init()
    log.info("Instantiating loggers...")
    # logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
    wandb_logger = hydra.utils.instantiate(cfg.logger.wandb)

    if cfg.get("torch_matmul_precision"):
        torch.set_float32_matmul_precision(cfg.torch_matmul_precision)
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    if cfg.get("distance"):
        verification_head_args = {'distance': hydra.utils.instantiate(cfg.distance)}
    else:
        verification_head_args = {}
    verification_head = hydra.utils.instantiate(cfg.verification_head, **verification_head_args)
    wandb.config.update(OmegaConf.to_container(cfg))

    datamodule: Optional[BaseDatamodule] = setup_datamodule(cfg.datamodule)

    net: Module = setup_model(cfg.model, datamodule)
    model: LightningModule = setup_lightning_module(cfg, datamodule, net, verification_head)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = initialize_callbacks(cfg)


    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=wandb_logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": wandb_logger,
        "trainer": trainer,
    }

    utils.log_hyperparameters(
        cfg=cfg,
        model=model,
        trainer=trainer,
    )

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    upload_run_dir_to_wandb(cfg)

    if cfg.get("validate_before_train"):
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        upload_run_dir_to_wandb(cfg)

    train_metrics = trainer.callback_metrics

    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     ckpt_path = trainer.checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning("Best ckpt not found! Using current weights for testing...")
    #         ckpt_path = None
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #     log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


def upload_run_dir_to_wandb(cfg):
    for file in pathlib.Path(cfg.paths.output_dir).glob("**/*"):
        wandb.save(str(file), base_path=cfg.paths.output_dir)


def trap_signals_to_handle_kills_gracefully():
    def handle_kill(signal_number, _frame):
        log.warning("getting killed")
        try:
            import wandb

            wandb.mark_preempting()
            wandb.finish(exit_code=-1)
        except Exception as e:
            log.error("an error occurred during training abort:")
            log.error(e)

        signal_name = {int(s): s for s in signal.Signals}[signal_number].name
        log.warning(f"aborting because of {signal_name} signal")
        raise SystemExit(f"aborting because of {signal_name} signal")

    signal.signal(signal.SIGINT, handle_kill)
    signal.signal(signal.SIGTERM, handle_kill)


if __name__ == "__main__":
    trap_signals_to_handle_kills_gracefully()
    main()

