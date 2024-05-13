from pathlib import Path
from typing import Optional, List

import hydra
import wandb
from lightning import LightningModule, Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.nn import Module

from src import utils
from src.datamodules.base_datamodule import BaseDatamodule
from src.verification_heads import VerificationHeadBase
from src.callbacks.log_best_metrics_callback import LogBestMetrics
from src.callbacks.user_halt_callback import UserHaltCallback

log = utils.get_pylogger(__name__)


def setup_lightning_module(config, datamodule: BaseDatamodule, model: Module, verification_head: VerificationHeadBase):
    log.info(f"Instantiating lightning module <{config.lightning_module._target_}>")
    identifier: LightningModule = hydra.utils.instantiate(
        config.lightning_module,
        model=model,
        metrics=config.monitored_metrics.metrics,
        klasses=(datamodule.train_dataset or datamodule.test_dataset).klasses,
        verification_head=verification_head
    )

    return identifier


def setup_datamodule(datamodule_config, stage="train", limit_files=None) -> Optional[BaseDatamodule]:
    log.info(f"Instantiating datamodule <{datamodule_config._target_}>")
    datamodule: Optional[BaseDatamodule] = hydra.utils.instantiate(datamodule_config)
    if datamodule:
        datamodule.setup(stage, limit_files)
    return datamodule


def setup_model(model_config, datamodule: BaseDatamodule):
    log.info(f"Instantiating model <{model_config._target_}>")
    dataset = datamodule.train_dataset or datamodule.test_dataset
    if model_config.num_out_classes == "auto":
        model_config.num_out_classes = dataset.num_classes
    model: LightningModule = hydra.utils.instantiate(
        model_config, num_features=dataset.num_features
    )
    log.info(str(model))
    return model


def initialize_callbacks(cfg) -> List[Callback]:
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.debug(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    if "monitored_callbacks" in cfg:

        def build_checkpoint_callback_list(checkpoint_metrics, store_path):
            checkpoint_metric_callbacks = []
            for metric in checkpoint_metrics:
                metric_id = "{name}_{dataset}".format(**metric)

                checkpoint_metric_callbacks.append(
                    ModelCheckpoint(
                        monitor=metric_id,
                        save_top_k=1,
                        mode=metric["mode"],
                        dirpath=store_path,
                        filename=f"{metric['mode']}_{metric_id}",
                    )
                )
            return checkpoint_metric_callbacks

        callbacks += build_checkpoint_callback_list(
            cfg.monitored_metrics.metrics, store_path="early_stopping"
        )

    callbacks += [LogBestMetrics(), UserHaltCallback()]
    return callbacks


def get_hydra_config_from_folder(folder: Path):
    checkpoint = folder.joinpath("model.ckpt")
    # override = OmegaConf.load(folder.joinpath(".hydra/overrides.yaml"))
    hydra_config = OmegaConf.load(folder.joinpath("actual-config.yaml"))
    hydra_config.datamodule.data_stats_path = folder.joinpath("train_stats.json")
    hydra_config.ckpt_path = checkpoint
    history_config = OmegaConf.load(folder.joinpath("history-config.yaml"))
    if "Mahalanobis" in hydra_config.verification_head._target_:
        hydra_config.verification_head.threshold = history_config['TAR_threshold/1/TAR@FAR=0.001%/validation']
    else:
        hydra_config.verification_head.threshold = history_config['TAR_threshold/10/TAR@FAR=0.001%/validation']
    # for override_entry in override:
    #     key, value = override_entry.split("=")
    #     if key == "experiment":
    #         continue
    #     value_to_override = hydra_config
    #     key_splits = key.split(".")
    #     for dict_key in key_splits[:-1]:
    #         value_to_override = value_to_override[dict_key]
    #     value_to_override[key_splits[-1]] = value
    return hydra_config


def download_checkpoint_with_run_config(wandb_run: wandb.apis.public.Run, model_output_dir: Path):
    try:
        files = {file.name: file for file in wandb_run.files()}
        hydra_files = {key: value for key, value in files.items() if ".hydra/" in key}
        train_stats_file = files['train_stats.json']
        # checkpoint = [file for name, file in files.items() if "max_precision_at_1" in name][0]
        best_epoch = sorted(
            [hist_dict for hist_dict in wandb_run._full_history(1_000_000_000)
             if hist_dict['validation_computation_time'] is not None and hist_dict["epoch"] != 0],
            key=lambda x: x['precision_at_1/validation'], reverse=True)[0]
        all_artifacts = wandb.Api().artifact_versions(name=f"99fe99/deep-vr-user-verification/model-{wandb_run.id}",
                                                      type_name="model")
        checkpoints = [
            version for version in all_artifacts
            if f"{best_epoch['epoch']}_max_precision_at_1" in version.metadata.get("original_filename")]
        if len(checkpoints) != 1:
            raise ValueError(f"Run {wandb_run.id} has {len(checkpoints)} checkpoints")
        checkpoint = checkpoints[0].files()[0]
        history_config = OmegaConf.create(best_epoch)

        root = model_output_dir.joinpath(wandb_run.id)
        root.mkdir(parents=True, exist_ok=True)
        print(f"Saving to <{root.absolute()}>")
        train_stats_file.download(root, replace=True)
        [hydra_file.download(root, replace=True) for hydra_file in hydra_files.values()]
        checkpoint.download(root, replace=True)
        OmegaConf.save(history_config, root.joinpath("history-config.yaml"))
        OmegaConf.save(OmegaConf.create(wandb_run.config), root.joinpath("actual-config.yaml"))
        return root.name
    except Exception as exception:
        print(f"Exception for run <{wandb_run.id}, {wandb_run.name}>:", exception)
        return None
