from dataclasses import dataclass
from typing import Optional

from src.data_encoding import DataEncoding


@dataclass
class DataHyperparameters:
    data_encoding: DataEncoding

    def __post_init__(self) -> None:
        if isinstance(self.data_encoding, str):
            self.data_encoding = DataEncoding(self.data_encoding)


@dataclass
class WindowDataHyperparameters(DataHyperparameters):
    fps: int
    window_size: int  # in frames
    original_fps: int
    displace_positional_data: Optional[bool] = False
    max_number_of_frames_per_user: Optional[int] = None
    use_6d_rotation: bool = False

    @property
    def frame_step_size(self):
        return self.original_fps // self.fps

    @property
    def window_length_in_s(self):
        return self.window_size / self.fps

    @property
    def token(self):
        return f"{self.data_encoding}_{self.fps}fps_{self.window_size}window-size"

