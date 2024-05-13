import pathlib
import random
from typing import List

import numpy as np
from torch.utils.data import get_worker_info, IterableDataset

from src.datamodules.datasets.window_dataset_for_scaled_data import WindowDataset
from src.utils import utils

log = utils.get_logger(__name__)


# based on https://math.stackexchange.com/q/2971563
def combined_mean_std(ns, stats: np.ndarray):
    assert len(stats.shape) == 3
    assert stats.shape[1] == 2

    ns = ns[:, None]
    means = stats[:, 0]
    stds = stats[:, 1]

    combined_mean = (ns * means).sum(axis=0) / ns.sum()

    combined_variance = (
                                (ns * stds ** 2).sum(axis=0) + (ns * (means - combined_mean) ** 2).sum(axis=0)
                        ) / ns.sum()
    combined_std = np.sqrt(combined_variance)

    return combined_mean, combined_std


class ChainedWindowDataset(IterableDataset):
    def __init__(
            self,
            data_files: List[pathlib.Path],
            max_files_to_load_at_once,
            shuffle=False,
            enforce_data_stats=None,
            **dataset_kwargs,
    ):
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(data_files)
        self.data_files = data_files
        self.max_files_to_load_at_once = max_files_to_load_at_once
        self.enforce_data_stats = enforce_data_stats
        self.dataset_kwargs = dataset_kwargs
        self.evaluation_mode = False

        if enforce_data_stats:
            self._data_stats = enforce_data_stats

    def _setup_dset(self, files, data_stats=None):
        dset = WindowDataset(
            files,
            shuffle=self.shuffle,
            enforce_data_stats=data_stats,
            **self.dataset_kwargs,
        )
        dset.evaluation_mode = self.evaluation_mode
        return dset

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            files_per_worker = self.max_files_to_load_at_once
            chunks = np.array_split(self.data_files, round(len(self.data_files) / files_per_worker))
            for chunk_of_files in chunks:
                dset = self._setup_dset(chunk_of_files, data_stats=self.data_stats)
                sample_ids = np.arange(len(dset))

                if self.shuffle:
                    np.random.shuffle(sample_ids)

                for sample_id in sample_ids:
                    yield dset[sample_id]
            del dset, sample_ids
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            assert self.max_files_to_load_at_once >= num_workers

            files_per_worker = round(self.max_files_to_load_at_once / num_workers)

            chunks = np.array_split(self.data_files, max(round(len(self.data_files) / files_per_worker), num_workers))

            target_chunk_idxs = np.where((np.arange(len(chunks)) % num_workers) == worker_id)[0]

            for chunk_idx in target_chunk_idxs:
                dset = self._setup_dset(chunks[chunk_idx], data_stats=self.data_stats)

                sample_ids = np.arange(len(dset))

                if self.shuffle:
                    np.random.shuffle(sample_ids)

                for sample_id in sample_ids:
                    yield dset[sample_id]
            del dset

    @property
    def data_stats(self):
        assert hasattr(self, "_data_stats"), "Data stats should be set in the base datamodule"
        return self._data_stats

    @property
    def num_classes(self):
        return len(self.data_stats["klasses"])

    @property
    def num_features(self):
        return len(self.data_stats["means"])

    @property
    def klasses(self):
        return self.data_stats["klasses"]

    @property
    def loss_weights(self):
        raise NotImplementedError("yet to be implemented")
