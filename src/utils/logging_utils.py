import os
import lightning as L
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf, DictConfig

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def log_hyperparameters(
        cfg: DictConfig,
        model: L.LightningModule,
        trainer: L.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = cfg["trainer"]
    hparams["model"] = cfg["model"]
    hparams["datamodule"] = cfg["datamodule"]

    hparams["paths"] = cfg["paths"]

    hparams["optimizer_options"] = cfg["lightning_module"].get(
        "optimizer_options", None
    )
    hparams["loss_options"] = cfg["lightning_module"].get("loss_options", None)

    if "temperature" in hparams:
        hparams["temperature"] = cfg["lightning_module"]["temperature"]

    if "seed" in cfg:
        hparams["seed"] = cfg["seed"]
    # if "callbacks" in config:
    #     hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model.params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model.params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model.params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    if node_name := os.environ.get("NODE_NAME"):
        hparams["node_name"] = node_name

    trainer.logger.log_hyperparams({k.replace("/", "."): v for k, v in hparams.items()})

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = lambda _: None
