import json
from collections.abc import Mapping
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

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
            skip_first_frames: int = 0,
            **kwargs,
    ):
        self.data_files = data_files
        self.enforce_data_stats = enforce_data_stats
        self.take_every_nth_sample = take_every_nth_sample
        self.window_size = window_size
        self.evaluation_mode = evaluation_mode
        self.skip_first_frames = skip_first_frames

        self._load_and_set_data()
        self._load_window_maker()

        self._set_or_load_data_stats()

        self._scale_data()

    def _load_window_maker(self):
        self.change_idxs = compute_change_idxs(self.take_id)
        num_frames = len(self.take_id)

        self.wm: WindowMaker = WindowMaker(
            num_frames,
            self.window_size,
            self.change_idxs,
            skip_first_frames=self.skip_first_frames,
            take_every_nth_sample=self.take_every_nth_sample,
        )

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

    def _set_or_load_data_stats(self):
        if self.enforce_data_stats:
            if isinstance(self.enforce_data_stats, Mapping):
                stats = self.enforce_data_stats
            else:
                stats = self.load_settings(self.enforce_data_stats)

            self.means = stats["means"]
            self.stds = stats["stds"]
            # self._klasses = np.sort(np.unique(stats["klasses"]))
            # self.encode_ground_truth_to_targets()
        else:
            self.means = self._own_means
            self.stds = self._own_stds

    def load_settings(self, path):
        settings = json.load(open(path, "r"))

        return {
            "means": np.array(settings["means"]),
            "stds": np.array(settings["stds"]),
            "klasses": np.array(settings["klasses"]),
        }

    @property
    def klasses(self) -> np.ndarray:
        if not hasattr(self, "_klasses"):
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

    def _scale_data(self):
        log.info("Scaling data")
        half_length = len(self.frames) // 2

        for index, column in enumerate(self.frames.columns):
            self.frames.iloc[:half_length, self.frames.columns.get_loc(column)] = (
                    (self.frames.iloc[:half_length, self.frames.columns.get_loc(column)] - self.means[index]) /
                    self.stds[index]
            ).copy()

            self.frames.iloc[half_length:, self.frames.columns.get_loc(column)] = (
                    (self.frames.iloc[half_length:, self.frames.columns.get_loc(column)] - self.means[index]) /
                    self.stds[index]
            ).copy()

            # self.frames[col] = (self.frames[col] - self.means[index]) / self.stds[index]
        log.info("Finished data")

        # self.frames = (self.frames - self.means) / self.stds

    @property
    def num_classes(self):
        return len(self.klasses)

    def _load_and_set_data(self):
        assert len(self.feature_columns) % 3 == 0

        # def load_data_file(file_path):
        #     if not file_path.exists():
        #         raise FileNotFoundError(f"could not locate file {file_path}, aborting...")
        #
        #     df = pd.read_feather(file_path).rename(columns=lambda col: col.replace("delta_", ""))
        #     float64_cols = df.select_dtypes(include="float64").columns.tolist()
        #     float16_cols = df.select_dtypes(include="float16").columns.tolist()
        #     int64_cols = df.select_dtypes(include="int64").columns.tolist()
        #
        #     mapper = ({col_name: np.float32 for col_name in float64_cols + float16_cols} |
        #               {col_name: np.int32 for col_name in int64_cols})
        #     df = df.astype(mapper)
        #     return df
        #
        # log.info(f"loading {len(self.data_files)} data files from disk")
        # data_df = (load_data_file(self.data_files[0]))
        # for file_path in tqdm(self.data_files[1:]):
        #     file_data = (load_data_file(file_path))
        #     data_df = pd.concat([data_df, file_data])
        #     data_df.reset_index(inplace=True, drop=True)
        def read_from_polars_to_pandas():
            try:
                log.info("Starting to read data")
                all_parts = []
                first_part_files = self.data_files[:len(self.data_files) // 2]
                if len(first_part_files) > 0:
                    first_half = pl.scan_ipc(first_part_files, memory_map=False,
                                             rechunk=False).collect()
                    all_parts.append(first_half.to_pandas())
                    log.info("First half done")

                second_part_files = self.data_files[len(self.data_files) // 2:]
                if len(second_part_files) > 0:
                    second_half = pl.scan_ipc(second_part_files, memory_map=False,
                                              rechunk=False).collect()
                    all_parts.append(second_half.to_pandas())
                    log.info("Second half done")
                return pd.concat(all_parts)
            except Exception as e:
                log.exception(e)
                raise e

        data_df = read_from_polars_to_pandas()
        data_df.reset_index(drop=True, inplace=True)
        # data_df = pl.scan_ipc(self.data_files, memory_map=False, rechunk=False).collect().to_pandas()

        log.info("Starting to intersect")
        feature_columns = np.intersect1d(self.feature_columns, data_df.columns)
        log.info(f"there are {len(feature_columns)} features")
        self.frames = data_df[feature_columns]

        log.info("computing stats")
        stds = []
        means = []
        for col in self.frames.columns:
            means.append(np.nanmean(self.frames[col]))
            stds.append(np.nanstd(self.frames[col]))
        self._own_means = np.array(means)
        self._own_stds = np.array(stds)

        log.info("misc")
        self.ground_truth = data_df.user_id
        if "session_idx" in data_df:
            self.session_idx = data_df.session_idx
        if "word" in data_df:
            self.words = data_df.word
            data_df.is_drawing[data_df.is_drawing.isna()] = False
            self.is_drawing = data_df.is_drawing.astype(int)
            self.drawing_hands = data_df.drawing_hand
            if "iteration" in data_df:
                self.iterations = data_df.iteration.astype(int)
                self.successful_takes = data_df.successful_take.astype(int)
        self.frame_idx = data_df.frame_idx
        self.take_id = data_df.take_id
        self.frames.reset_index(drop=True, inplace=True)
        assert len(self.frames.index) == len(np.unique(self.frames.index))

        log.info("start ground truth encoding")
        self.encode_ground_truth_to_targets()

        log.info("finished all data preprocessing")

    def encode_ground_truth_to_targets(self):
        mapping = self.klass_idx_mapping
        self.targets = np.array([mapping[y] for y in self.ground_truth], dtype="int16")

    @property
    def data_stats(self):
        return {"means": self.means, "stds": self.stds, "klasses": self.klasses}

    @property
    def raw_data_stats(self):
        return len(self), self._own_means, self._own_stds, self.klasses

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

        if hasattr(self, 'words'):
            item['words'] = self.words[frame_id]
            item['is_drawing'] = self.is_drawing[frame_id]
            item['drawing_hands'] = self.drawing_hands[frame_id]
            if hasattr(self, "iterations"):
                item['iteration'] = self.iterations[frame_id]
                item['successful_takes'] = self.successful_takes[frame_id]

        return item
