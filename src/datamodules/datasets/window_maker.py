# %%

from typing import Union

import pandas as pd
import numpy as np


class WindowMaker:
    def __init__(self, num_frames, window_size, change_idxs, skip_first_frames: int, take_every_nth_sample: int):
        self.skip_first_frames = skip_first_frames
        self.num_frames = num_frames
        self.window_size = window_size
        self.change_idxs = change_idxs
        self.take_every_nth_sample = take_every_nth_sample
        self.frame_ids = self._compute_valid_frame_ids()

    @property
    def num_windows(self):
        return len(self.frame_ids)

    @property
    def num_samples(self):
        return self.num_windows

    def _compute_valid_frame_ids(self):
        def valid_frame_id_generator():
            change_idxs_incl_end = list(self.change_idxs) + [self.num_frames - 1]
            for start_change_idx_idx in range(len(change_idxs_incl_end) - 1):
                start_idx = change_idxs_incl_end[start_change_idx_idx] + self.skip_first_frames
                end_idx = change_idxs_incl_end[start_change_idx_idx+1] - self.window_size + 1
                yield np.arange(start_idx, end_idx, self.take_every_nth_sample, dtype="uint32")

        valid_frame_ids = np.concatenate(list(valid_frame_id_generator()))

        return valid_frame_ids

    def sample_to_frame_id(self, sample_id):
        return self.frame_ids[sample_id]

    def get_window(self, data: Union[pd.DataFrame, np.ndarray], frame_id: int) -> np.ndarray:
        window = data[frame_id:frame_id+self.window_size]

        return window

    @staticmethod
    def compute_change_idxs(values):
        return np.where(np.diff(values, prepend=np.nan))[0]

