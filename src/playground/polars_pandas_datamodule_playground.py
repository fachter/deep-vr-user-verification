import pathlib
import random

import polars as pl
import pandas as pd
import numpy as np
import os
from time import time
from tqdm import tqdm

# %%

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
print(os.getcwd())
random.seed(12345)
data_files = sorted(pathlib.Path("data/boxrr-BRA-50_fps-600").glob("*.ftr"))
random.shuffle(data_files)
selected_files = data_files[:20]


# %%
def load_data_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"could not locate file {file_path}, aborting...")

    df = pd.read_feather(file_path).rename(columns=lambda col: col.replace("delta_", ""))
    float64_cols = df.select_dtypes(include="float64").columns.tolist()
    float16_cols = df.select_dtypes(include="float16").columns.tolist()
    int64_cols = df.select_dtypes(include="int64").columns.tolist()

    mapper = ({col_name: np.float32 for col_name in float64_cols + float16_cols} |
              {col_name: np.int32 for col_name in int64_cols})
    df = df.astype(mapper)
    return df


start = time()
polars_data = pl.scan_ipc(selected_files, memory_map=False)
print(polars_data.schema)
polars_data = polars_data.collect().to_pandas()
end = time()
print(f"Polars {end - start:5f}")
start = time()
data_df = (load_data_file(selected_files[0]))
for file_path in tqdm(selected_files[1:]):
    file_data = (load_data_file(file_path))
    data_df = pd.concat([data_df, file_data])
    data_df.reset_index(inplace=True, drop=True)
end = time()
print(f"Pandas {end - start:5f}")

# %%
print(len(df))


# %%

def load_and_set_data():
    assert len(feature_columns) % 3 == 0

    def load_data_file(file_path):
        if not file_path.exists():
            raise FileNotFoundError(f"could not locate file {file_path}, aborting...")

        df = pd.read_feather(file_path).rename(columns=lambda col: col.replace("delta_", ""))
        float64_cols = df.select_dtypes(include="float64").columns.tolist()
        float16_cols = df.select_dtypes(include="float16").columns.tolist()
        int64_cols = df.select_dtypes(include="int64").columns.tolist()

        mapper = ({col_name: np.float32 for col_name in float64_cols + float16_cols} |
                  {col_name: np.int32 for col_name in int64_cols})
        df = df.astype(mapper)
        return df

    log.info(f"loading {len(data_files)} data files from disk")
    data_df = (load_data_file(data_files[0]))
    for file_path in tqdm(data_files[1:]):
        file_data = (load_data_file(file_path))
        data_df = pd.concat([data_df, file_data])
        data_df.reset_index(inplace=True, drop=True)

    feature_columns = np.intersect1d(feature_columns, data_df.columns)
    log.info(f"there are {len(feature_columns)} features")
    frames = data_df[feature_columns]

    log.info("computing stats")
    stds = []
    means = []
    for col in frames.columns:
        means.append(np.nanmean(frames[col]))
        stds.append(np.nanstd(frames[col]))
    _own_means = np.array(means)
    _own_stds = np.array(stds)

    log.info("misc")
    ground_truth = data_df.user_id
    if "session_idx" in data_df:
        session_idx = data_df.session_idx
    frame_idx = data_df.frame_idx
    take_id = data_df.take_id
    frames.reset_index(drop=True, inplace=True)
    assert len(frames.index) == len(np.unique(frames.index))

    log.info("start ground truth encoding")
    self.encode_ground_truth_to_targets()

    log.info("finished all data preprocessing")
