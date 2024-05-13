import torch
import numpy as np
from typing import List
import random
import pathlib

from torch.utils.data import IterableDataset, Dataset, DataLoader, get_worker_info
import os
import pandas as pd
import polars as pl
from tqdm import tqdm

from src.datamodules.datasets.window_dataset_for_scaled_data import WindowDataset
from src.datamodules.datasets.chained_window_dataset import ChainedWindowDataset, combined_mean_std
from src.playground.move_data_to_sub_folders_and_scale_them import get_stats_from_files

# %%
print(os.getcwd())
# data_path = pathlib.Path("../../../bra_50_fps-600_users/train")
data_path = pathlib.Path("./data/boxrr-BRA-50_fps-600")
print(data_path.exists())
files = sorted(data_path.glob("*.ftr"))
random.seed(12345)
random.shuffle(files)
print(len(files))
# %%

data_stats = {
    'means': np.array([7.14217039e-01, -1.42537861e-02, 8.70764237e-03, -2.11554270e-03,
                       -2.87557090e+01, -4.08171566e+01, -1.78104551e+01, 5.93253895e-01,
                       4.01989523e-02, 2.36302544e-02, -1.13137589e-01, 2.78070239e+01,
                       -4.06152771e+01, -1.88837908e+01, 5.87402935e-01, 3.44903381e-02,
                       -1.30240058e-02, 1.13440923e-01]),
    'stds': np.array([0.68659735, 0.08573431, 0.09252638, 0.04778209, 18.2595538,
                      34.66640669, 19.91964523, 0.22832713, 0.54186949, 0.40149584,
                      0.35513465, 19.07725076, 33.6573646, 19.86841373, 0.23057869,
                      0.5392024, 0.41704302, 0.35056968]),
    'klasses': np.array(files)
}
dset = WindowDataset(
    files[:5], shuffle=True,
    window_size=1500,
    take_every_nth_sample=3000,
    enforce_data_stats=data_stats
)

# %%
print(len(dset))
# %%
counter = 0
targets = []
shuffled_dset = list(dset)
print("Created list")
random.shuffle(shuffled_dset)
print("shuffled list")
for value in shuffled_dset:
    counter += 1
    if counter > 10_000:
        break
    targets.append(value['targets'])
unique_values, unique_counts = np.unique(targets, return_counts=True)
print(unique_values, unique_counts)
print(unique_counts.sum())
# %%
stats_train = get_stats_from_files(files[:400])
# %%

stats_0_50 = get_stats_from_files(files[:50])
print("Done 0-50")
stats_50_100 = get_stats_from_files(files[50:100])
print("Done 50-100")
stats_100_150 = get_stats_from_files(files[100:150])
print("Done 100-150")
stats_150_200 = get_stats_from_files(files[150:200])
print("Done 150-200")
stats_200_250 = get_stats_from_files(files[200:250])
print("Done 200-250")
stats_250_300 = get_stats_from_files(files[250:300])
print("Done 250-300")

# %%
all_stats = [
    stats_0_50, stats_50_100, stats_100_150, stats_150_200, stats_200_250, stats_250_300
]
all_stats = np.array(all_stats)

means_row = all_stats[:, 0, None, :]
means_col = all_stats[None, :, 0, :]
std_row = all_stats[:, 1, None, :]
std_col = all_stats[None, :, 1, :]

means_diff = means_row - means_col
std_diff = std_row - std_col

print(means_diff.mean(axis=(0, 1)))
print(std_diff.mean(axis=(0, 1)))
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


def get_stats_from_file(file):
    df = load_df_from_file(file)
    df.reset_index(inplace=True, drop=True)
    fc = np.intersect1d(feature_columns, df.columns)
    frames = df[fc]
    means = np.nanmean(frames, axis=0)
    stds = np.nanstd(frames, axis=0)
    length = len(frames)
    unique_gt = df.user_id.unique_values, unique_counts()
    return {
        'mean': means,
        'std': stds,
        'length': length,
        'klasses': unique_gt
    }


def load_df_from_file(file):
    df = pd.read_feather(file)
    float64_cols = df.select_dtypes(include="float64").columns.tolist()
    float16_cols = df.select_dtypes(include="float16").columns.tolist()
    int64_cols = df.select_dtypes(include="int64").columns.tolist()
    mapper = ({col_name: np.float32 for col_name in float64_cols + float16_cols} |
              {col_name: np.int32 for col_name in int64_cols})
    df = df.astype(mapper)
    return df


mean_values = []
std_values = []
length_values = []
unique_gt = []
for file in tqdm(files):
    stats = get_stats_from_file(file)
    mean_values.append(stats['mean'])
    std_values.append(stats['std'])
    length_values.append(stats['length'])
    unique_gt.extend(stats['klasses'])

mean_std_stats = np.array(list(zip(mean_values, std_values)))
ns = np.array(length_values)

result = combined_mean_std(ns, mean_std_stats)
print(result)

# %%
data_df = load_df_from_file(sorted(files)[0])
for file in sorted(files)[1:5]:
    file_data = load_df_from_file(file)
    data_df = pd.concat([data_df, file_data])
    data_df.reset_index(inplace=True, drop=True)
fc = np.intersect1d(feature_columns, data_df.columns)
frames = data_df[fc]
means = np.nanmean(frames, axis=0)
stds = np.nanstd(frames, axis=0)
unique_users = data_df.user_id.unique_values, unique_counts()

users = unique_users.tolist()
users.extend(unique_gt)
# %%
worker_id = 0
num_workers = 2
max_files_to_load_at_once = 10

chunks = np.array_split(files, round(len(files) / max_files_to_load_at_once))
chunks_per_worker = np.array_split(files, round(len(files) / round(max_files_to_load_at_once / num_workers)))
target_chunk_idxs = np.where((np.arange(len(chunks)) % num_workers) == worker_id)[0]
print(chunks_per_worker[target_chunk_idxs[0]])

# %%
users = set(users)


# %%
#
# dataset = ChainedWindowDataset(
#     files[:40], max_files_to_load_at_once=10,
#     data_hyperparameters={},
#     window_size=150,
#     take_every_nth_sample=1500
# )
#
# # %%
#
# dataloader = DataLoader(dataset, num_workers=0)
#
# for value in dataloader:
#     print(len(value))


# %%
# data_files = files[:40]
# max_files_to_load_at_once = 10
#
#
# print("collecting information over all data files")
# ns = []
# stats = []
# klasses = []
# sample_shapes = set()
# num_samples = 0
#
# chunks = np.array_split(data_files, np.ceil(len(data_files) / max_files_to_load_at_once))
#
# for dset_idx, chunk in enumerate(chunks):
#     dset = WindowDataset(
#         chunk,
#         window_size=150,
#         take_every_nth_sample=1500,
#         enforce_data_stats=None
#     )
#     print(f" loading train chunk #{dset_idx:02d}")
#     n, means, stds, dset_klasses = dset.raw_data_stats
#     ns.append(n)
#     stats.append((means, stds))
#     klasses += dset_klasses.tolist()
#     sample_shapes.add(dset.sample_shape)
#     num_samples += len(dset)
#
# assert len(sample_shapes) == 1
# combined_mean, combined_std = combined_mean_std(np.array(ns), np.array(stats))
#
# self.sample_shape = sample_shapes.pop()
#
# self._data_stats = {
#     "n": sum(ns),
#     "means": combined_mean,
#     "stds": combined_std,
#     "klasses": klasses
# }
#
#

# %%

class SimpleIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        print("called iter")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            per_worker = int(np.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))


ds = SimpleIterableDataset(10, 100)
dataloader = DataLoader(ds, num_workers=0, batch_size=10)
print(list(dataloader))
dataloader = DataLoader(ds, num_workers=2, batch_size=10)
print(list(dataloader))

# %%
# %%
