import json
from collections.abc import Mapping
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.datamodules.datasets.helpers import compute_change_idxs
from src.datamodules.datasets.window_maker import WindowMaker
from src.utils import utils
from tqdm import tqdm

log = utils.get_logger("WindowDataset")


class WindowDataset:
    feature_columns = [
        "head_pos_x",
        "head_pos_y",
        "head_pos_z",
        "head_rot_x",
        "head_rot_y",
        "head_rot_z",
        "head_rot_w",
        "left_hand_pos_x",
        "left_hand_pos_y",
        "left_hand_pos_z",
        "left_hand_rot_x",
        "left_hand_rot_y",
        "left_hand_rot_z",
        "left_hand_rot_w",
        "right_hand_pos_x",
        "right_hand_pos_y",
        "right_hand_pos_z",
        "right_hand_rot_x",
        "right_hand_rot_y",
        "right_hand_rot_z",
        "right_hand_rot_w",
    ]

    frame_step_size: int = "unset"
    frames: pd.DataFrame = None

    def __init__(
        self,
        data_files: List[pathlib.Path],
        window_size: int,
        enforce_data_stats: Optional[Dict[str, Any]] = None,
        take_every_nth_sample=None,
        evaluation_mode: bool = False,
        klasses: List[str] = None,
        **kwargs,
    ):
        self.data_files = data_files
        self.enforce_data_stats = enforce_data_stats
        self.take_every_nth_sample = take_every_nth_sample
        self.window_size = window_size
        self.evaluation_mode = evaluation_mode

        self._klasses = klasses

        self._load_and_set_data()
        self._load_window_maker()

        self.means = enforce_data_stats['means']
        self.stds = enforce_data_stats['stds']
        self._scale_data()

    def _load_window_maker(self):
        self.change_idxs = compute_change_idxs(self.take_id)
        num_frames = len(self.take_id)

        self.wm: WindowMaker = WindowMaker(
            num_frames,
            self.window_size,
            self.change_idxs,
            skip_first_frames=50,
            take_every_nth_sample=self.take_every_nth_sample,
        )

    def _scale_data(self):
        data = pd.DataFrame()
        for index, col in enumerate(self.frames.columns):
            data[col] = (self.frames[col] - self.means[index]) / self.stds[index]
        self.frames = data

    @property
    def frame_ids(self):
        return self.wm.frame_ids

    @property
    def num_features(self):
        return self.frames.shape[1]

    @property
    def num_samples(self):
        return self.wm.num_windows

    @property
    def sample_shape(self) -> Tuple[int, int]:
        return self.window_size, self.num_features

    # def load_settings(self, path):
    #     settings = json.load(open(path, "r"))
    #
    #     return {
    #         "means": np.array(settings["means"]),
    #         "stds": np.array(settings["stds"]),
    #         "klasses": np.array(settings["klasses"]),
    #     }

    @property
    def klasses(self) -> np.ndarray:
        if self._klasses is None:
            self._klasses = np.sort(pd.unique(self.ground_truth))

        return self._klasses

    @property
    def klass_idx_mapping(self) -> Dict[str, int]:
        return {klass: idx for idx, klass in enumerate(self.klasses)}

    @property
    def loss_weights(self):
        if not hasattr(self, "_loss_weights"):
            target_counts = np.bincount(self.ground_truth)
            loss_weights = (target_counts / target_counts.max()) ** -1
            self._loss_weights = loss_weights / loss_weights.max()
        return self._loss_weights

    @property
    def per_sample_class_weights(self):
        if not hasattr(self, "_per_sample_loss_weights"):
            self._per_sample_loss_weights = self.loss_weights[self.targets]

        return self._per_sample_loss_weights

    def __len__(self):
        return self.num_samples

    @property
    def num_classes(self):
        return len(self.klasses)

    def _load_and_set_data(self):
        assert len(self.feature_columns) % 3 == 0

        def load_data_file(file_path):
            if not file_path.exists():
                raise FileNotFoundError(f"could not locate file {file_path}, aborting...")

            df = pd.read_feather(file_path).rename(columns=lambda col: col.replace("delta_", ""))
            float64_cols = df.select_dtypes(include="float64").columns.tolist()
            float16_cols = df.select_dtypes(include="float16").columns.tolist()
            int64_cols = df.select_dtypes(include="int64").columns.tolist()

            mapper = {**{col_name: np.float32 for col_name in float64_cols + float16_cols},
                      **{col_name: np.int32 for col_name in int64_cols}}
            df = df.astype(mapper)
            return df

        log.info(f"loading {len(self.data_files)} data files from disk")
        data_df = (load_data_file(self.data_files[0]))
        for file_path in tqdm(self.data_files):
            file_data = (load_data_file(file_path))
            data_df = pd.concat([data_df, file_data])
            data_df.reset_index(inplace=True, drop=True)

        feature_columns = np.intersect1d(self.feature_columns, data_df.columns)
        log.info(f"there are {len(feature_columns)} features")
        self.frames = data_df[feature_columns]
        # self._own_means = np.nanmean(self.frames, axis=0)
        # self._own_stds = np.nanstd(self.frames, axis=0)

        log.info("misc")
        self.ground_truth = data_df.user_id
        if "session_idx" in data_df:
            self.session_idx = data_df.session_idx
        self.frame_idx = data_df.frame_idx
        self.take_id = data_df.take_id
        assert len(self.frames.index) == len(np.unique(self.frames.index))

        log.info("start ground truth encoding")
        self.encode_ground_truth_to_targets()

        log.info("finished all data preprocessing")

    def encode_ground_truth_to_targets(self):
        mapping = self.klass_idx_mapping
        self.targets = np.array([mapping[y] for y in self.ground_truth], dtype="int16")

    # @property
    # def data_stats(self):
    #     return {"means": self.means, "stds": self.stds, "klasses": self.klasses}

    # @property
    # def raw_data_stats(self):
    #     return (len(self), self._own_means, self._own_stds, self.klasses)

    def __getitem__(self, sample_id):
        frame_id = self.wm.sample_to_frame_id(sample_id)
        data = np.asarray(self.wm.get_window(self.frames, frame_id=frame_id))
        target = self.targets[frame_id]

        if len(data) == 0:
            raise IndexError

        item = {"data": data, "targets": target}

        if self.evaluation_mode:
            item["frame_id"] = frame_id.astype("int32")
            item["take_id"] = self.take_id[frame_id]
            item["session_idx"] = self.session_idx[frame_id] if hasattr(self, "session_idx") else item["take_id"]

        return item
