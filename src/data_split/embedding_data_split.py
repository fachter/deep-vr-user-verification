# %%

import hashlib
import os.path
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_split.base_data_split import BaseDataSplit
from src.utils import utils

log = utils.get_logger(__name__)


def compute_dataset_checksum(directory):
    hash_object = hashlib.sha256()
    files = sorted(list(directory.glob("*.ftr")))
    for file_path in files:
        hash_object.update(str(file_path.stem).encode())

    hash_value = hash_object.hexdigest()

    return hash_value


class EmbeddingDataSplit(BaseDataSplit):
    def __init__(self, data_folder: str, train: int, validation: int, test: int, seed: int, checksum: str):
        src_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_folder = pathlib.Path(os.path.join(src_folder, data_folder))

        assert self.data_folder.exists(), f"data folder {self.data_folder} does not exist!"
        
        files = sorted(list(self.data_folder.glob("*.ftr")))
        num_files = len(files)
        computed_checksum = compute_dataset_checksum(self.data_folder)
        
        assert checksum == computed_checksum, f"Checksum for dataset does not match! Expected: `{checksum}`; actual: `{computed_checksum}`; num_files={num_files}"

        self.seed = seed
        user_files = np.array(files)
        user_idxs = np.arange(len(user_files))
        np.random.seed(self.seed)
        np.random.shuffle(user_idxs)

        self.split: Dict[str, List[pathlib.Path]] = {
            "train": user_files[user_idxs[:train]],
            "validation": user_files[user_idxs[train:train+validation]],
            "test": user_files[user_idxs[-test:]],
        }

        assert len(self.split["train"]) == train, f"train has the wrong number of entries: expected {train} entries, bot got {self.split['train']}"
        assert len(self.split["validation"]) == validation, f"validation has the wrong number of entries: expected {validation} entries, bot got {self.split['validation']}"
        assert len(self.split["test"]) == test, f"test has the wrong number of entries: expected {test} entries, bot got {self.split['test']}"



# eds = EmbeddingDataSplit("data/30_fps-1000_users", 10, 20, 30, 42)

# # %%
# eds.split
