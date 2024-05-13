import os
import pathlib
from typing import Dict, List

import numpy as np

from src.data_split.base_data_split import BaseDataSplit


class MotionDataSplit(BaseDataSplit):
    def __init__(self, data_folder: str, limit_test_files=None, remove_test_files=None):
        if remove_test_files is None:
            remove_test_files = []
        src_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_folder = pathlib.Path(os.path.join(src_folder, data_folder))
        files = sorted([file for file in self.data_folder.glob("*.ftr")
                        if limit_test_files is None or file.name in limit_test_files
                        and file.name not in remove_test_files])
        self.split: Dict[str, List[pathlib.Path]] = {
            "train": [],
            "validation": [],
            "test": np.array(files),
        }
