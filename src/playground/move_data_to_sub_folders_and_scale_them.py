import pathlib
import os
import random

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import argparse

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


def main(data_path_str: str):
    data_path = pathlib.Path(data_path_str)
    assert data_path.exists(), "data path does not exist!!"
    files = sorted([file for file in data_path.glob("*.ftr")])
    random.shuffle(files)
    train = files[:400]
    validation = files[400:500]
    test = files[500:600]
    print(f"Train: {len(train)}, Validation: {len(validation)}, Test: {len(test)}")

    # data_path.joinpath("train").mkdir(exist_ok=True)
    # data_path.joinpath("validation").mkdir(exist_ok=True)
    # data_path.joinpath("test").mkdir(exist_ok=True)
    #
    # train = [file.rename(data_path.joinpath("train").joinpath(file.name)) for file in tqdm(train, desc="Move train files")]
    # validation = [file.rename(data_path.joinpath("validation").joinpath(file.name)) for file in tqdm(validation, desc="Move validation files")]
    # test = [file.rename(data_path.joinpath("test").joinpath(file.name)) for file in tqdm(test, desc="Move test files")]

    print("Calculating mean and std")

    result = get_stats_from_files(train)
    means, stds = result
    stat_df = pd.DataFrame()
    stat_df['means'] = means
    stat_df['stds'] = stds
    stat_df.to_csv(data_path.joinpath("400_users.stats.csv"), index=False)

    print("Calculated and saved stats to '400_users_stats.csv'")

    # for file in tqdm(train + validation + test, desc="Scale all files"):
    #     df = load_df_from_file(file)
    #     fc = np.intersect1d(feature_columns, df.columns)
    #     for index, colum in enumerate(fc):
    #         df[colum] = (df[colum] - means[index]) / stds[index]
    #     df.to_feather(file)
    #     # increased filesize by more than 100MB (216MB -> 350MB)

    print("Done")


def get_stats_from_files(files):
    mean_values = []
    std_values = []
    length_values = []
    unique_gt = []
    for file in tqdm(files, desc="get stats for files"):
        stats = get_stats_from_file(file)
        mean_values.append(stats['mean'])
        std_values.append(stats['std'])
        length_values.append(stats['length'])
        unique_gt.extend(stats['klasses'])
    mean_std_stats = np.array(list(zip(mean_values, std_values)))
    result = combined_mean_std(np.array(length_values), mean_std_stats)
    return result


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




def get_stats_from_file(file):
    df = load_df_from_file(file)
    df.reset_index(inplace=True, drop=True)
    fc = np.intersect1d(feature_columns, df.columns)
    frames = df[fc]
    means = np.nanmean(frames, axis=0)
    stds = np.nanstd(frames, axis=0)
    length = len(frames)
    unique_gt = df.user_id.unique()
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


if __name__ == '__main__':
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    args = parser.parse_args()
    main(args.data_path)
