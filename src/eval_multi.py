import os
import datetime
import re
from typing import List, Tuple, Optional
from pathlib import Path

from lightning import LightningModule, Trainer
from omegaconf import OmegaConf, DictConfig
import torch
import lightning as L
import wandb
import hydra
import pyrootutils
import polars as pl
from torch.nn import Module


pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.lightning_modules.verification_embedding_module import VerificationEmbeddingModule
from src import utils
from src.datamodules.base_datamodule import BaseDatamodule
from src.setup_utils import setup_model, setup_datamodule, setup_lightning_module, download_checkpoint_with_run_config, \
    get_hydra_config_from_folder

log = utils.get_pylogger(__name__)


def evaluate(cfg: DictConfig) -> None:
    api = wandb.Api(
        overrides={
            'entity': '99fe99',
            'project': 'deep-vr-user-verification'
        }
    )
    if cfg.get("torch_matmul_precision"):
        torch.set_float32_matmul_precision(cfg.torch_matmul_precision)
    seed = cfg.get("seed")
    cfg.datamodule.data_stats_path = Path(cfg.paths.root_dir).joinpath(f'train_stats_{seed}.json')
    datamodule: Optional[BaseDatamodule] = setup_datamodule(cfg.datamodule, "test", 60)
    model_output_dir = Path(cfg.paths.root_dir).joinpath("model_outputs")
    for run_id in cfg.run_ids:
        if seed:
            L.seed_everything(cfg.seed, workers=True)
        run_name = download_checkpoint_with_run_config(
            api.run(f"99fe99/deep-vr-user-verification/{run_id}"),
            model_output_dir
        )
        if run_name is None:
            log.warning(f"Could not download checkpoint")
            continue
        folder = Path(Path(cfg.paths.root_dir).joinpath("model_outputs", run_name))
        model_hydra_config = get_hydra_config_from_folder(folder)
        if model_hydra_config.seed != seed:
            log.warning(f"Config seed {seed} does not match hydra")
            continue

        run_evaluation(datamodule, model_hydra_config, run_id)


def run_evaluation(datamodule, model_hydra_config, original_run_id):
    wandb.init(project="deep-vr-user-verification", entity="99fe99", group="test-step")
    wandb.log({'original_run_id': original_run_id})
    log.info("Instantiating loggers...")
    wandb_logger = hydra.utils.instantiate(model_hydra_config.logger.wandb)
    wandb.config.update(OmegaConf.to_container(model_hydra_config))

    # if model_hydra_config.get("distance"):
    #     verification_head_args = {'distance': hydra.utils.instantiate(model_hydra_config.distance)}
    # else:
    #     verification_head_args = {}
    # verification_head = hydra.utils.instantiate(model_hydra_config.verification_head, **verification_head_args)

    # net: Module = setup_model(model_hydra_config.model, datamodule)
    # model: LightningModule = setup_lightning_module(model_hydra_config, datamodule, net, verification_head)
    model = VerificationEmbeddingModule.load_from_checkpoint(model_hydra_config.ckpt_path)
    model.verification_head.best_threshold = model_hydra_config.verification_head.threshold

    trainer: Trainer = hydra.utils.instantiate(model_hydra_config.trainer, logger=wandb_logger)

    log.info("Finished instantiating / Starting evaluation")
    trainer.test(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path=model_hydra_config.ckpt_path)
    wandb.finish()


@hydra.main(config_path="../configs", version_base="1.3", config_name="eval_multi.yaml")
def main(cfg: DictConfig) -> None:
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == '__main__':
    main()
