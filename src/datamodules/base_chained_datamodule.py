import json
import os
import pathlib
from typing import Optional

import numpy as np
from lightning import LightningDataModule
from src.utils import utils

from src.data_split.base_data_split import BaseDataSplit
from src.datamodules.datasets.chained_window_dataset import ChainedWindowDataset

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))
# NUM_WORKERS = 0

log = utils.get_logger(__name__)


class BaseChainedDatamodule(LightningDataModule):
    train_dataset: ChainedWindowDataset = None
    val_dataset: ChainedWindowDataset = None
    test_dataset: ChainedWindowDataset = None

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
        self.data_stats = self.load_data_stats()

    def setup(self, stage: Optional[str] = "train"):
        if stage == "test":
            self.test_dataset = ChainedWindowDataset(
                self.data_split.split["test"],
                enforce_data_stats=self.data_stats,
                **{**self.common_dataset_kwargs, **self.test_dataset_kwargs}
            )
            self.test_dataset.evaluation_mode = True
        elif stage == "validate":
            self.val_dataset = self._load_val()
        elif stage == "train" or stage == "fit":
            if not self.train_dataset:
                self.train_dataset = ChainedWindowDataset(
                    self.data_split.split["train"],
                    enforce_data_stats=self.data_stats,
                    shuffle=True,
                    **{**self.common_dataset_kwargs, **self.train_dataset_kwargs}
                )
                # self.safe_train_data_stats_settings()

            if not self.val_dataset:
                self.val_dataset = self._load_val()

    def _load_val(self):
        return ChainedWindowDataset(
            self.data_split.split["validation"],
            enforce_data_stats=self.data_stats,
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
        # TODO: calculate scaling vectors with caching
        return {
            'means': np.array([7.14217039e-01, -1.42537861e-02, 8.70764237e-03, -2.11554270e-03,
                               -2.87557090e+01, -4.08171566e+01, -1.78104551e+01, 5.93253895e-01,
                               4.01989523e-02, 2.36302544e-02, -1.13137589e-01, 2.78070239e+01,
                               -4.06152771e+01, -1.88837908e+01, 5.87402935e-01, 3.44903381e-02,
                               -1.30240058e-02, 1.13440923e-01]),
            'stds': np.array([0.68659735, 0.08573431, 0.09252638, 0.04778209, 18.2595538,
                              34.66640669, 19.91964523, 0.22832713, 0.54186949, 0.40149584,
                              0.35513465, 19.07725076, 33.6573646, 19.86841373, 0.23057869,
                              0.5392024, 0.41704302, 0.35056968]),
            'klasses': np.array([file.name for file in list(self.data_split.split['train']) +
                                 list(self.data_split.split['validation']) + list(self.data_split.split['test'])])
        }

