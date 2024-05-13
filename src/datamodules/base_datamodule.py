import json
import os
import pathlib
from typing import Optional

from src.datamodules.datasets.window_dataset import WindowDataset
from lightning import LightningDataModule
from src.utils import utils

from src.data_split.base_data_split import BaseDataSplit

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))

log = utils.get_logger(__name__)
log.info(f"Using {NUM_WORKERS} workers")

class BaseDatamodule(LightningDataModule):
    train_dataset: WindowDataset = None
    val_dataset: WindowDataset = None
    test_dataset: WindowDataset = None

    default_dataloader_settings = dict(
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    def __init__(
        self,
        data_split: BaseDataSplit,
        batch_size: int,
        data_stats_path,
        data_loader_settings=None,
        common_dataset_kwargs=None,
        train_dataset_kwargs=None,
        validation_dataset_kwargs=None,
        test_dataset_kwargs=None,
    ):
        super().__init__()
        
        self.common_dataset_kwargs = common_dataset_kwargs if common_dataset_kwargs else {}
        self.train_dataset_kwargs = train_dataset_kwargs if train_dataset_kwargs else {}
        self.validation_dataset_kwargs = validation_dataset_kwargs if validation_dataset_kwargs else {}
        self.test_dataset_kwargs = test_dataset_kwargs if test_dataset_kwargs else {}

        self.data_stats_path = pathlib.Path(data_stats_path)

        if not data_loader_settings:
            data_loader_settings = {}

        self.data_loader_settings = {
            **self.default_dataloader_settings,
            **data_loader_settings,
            "batch_size": batch_size,
        }

        self.batch_size = batch_size
        self.data_split = data_split

    def setup(self, stage: Optional[str] = "train", limit_files=None):
        if stage == "test":
            files_to_use = self.data_split.split["test"]
            if limit_files is not None:
                files_to_use = files_to_use[-limit_files:]
            self.test_dataset = WindowDataset(
                files_to_use,
                enforce_data_stats=self.load_data_stats(),
                **{**self.common_dataset_kwargs, **self.test_dataset_kwargs}
            )
            self.test_dataset.evaluation_mode = True
        elif stage == "validate":
            self.val_dataset = self._load_val()
        elif stage == "train" or stage == "fit":
            if not self.train_dataset:
                self.train_dataset = WindowDataset(
                    self.data_split.split["train"],
                    shuffle=True,
                    **{**self.common_dataset_kwargs, **self.train_dataset_kwargs}
                )
                self.safe_train_data_stats_settings()

            if not self.val_dataset:
                self.val_dataset = self._load_val()

    def _load_val(self):
        return WindowDataset(
                self.data_split.split["validation"],
                enforce_data_stats=self.load_data_stats(),
                **{**self.common_dataset_kwargs, **self.validation_dataset_kwargs}
            )

    def safe_train_data_stats_settings(self):
        with open(self.data_stats_path, "w") as json_file:
            stats = self.train_dataset.data_stats.copy()
            stats["means"] = stats["means"].tolist()
            stats["stds"] = stats["stds"].tolist()
            stats["klasses"] = stats["klasses"].tolist()
            json.dump(
                stats,
                json_file,
            )
    
    def load_data_stats(self):
        with open(self.data_stats_path) as json_file:
            log.info(f"loading stats from {self.data_stats_path}")
            stats = json.load(
                json_file,
            )
        return stats