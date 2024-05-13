import pandas as pd
import os
from glob import glob
import pathlib
from dataclasses import dataclass
from typing import Dict, List

# %%
file_path = pathlib.Path("/Users/felixachter/source/uni/masterarbeit/half-life-alyx-data-preparation/data/output/wia"
                         "-30_fps-71_users")
files = sorted(file_path.glob("*.ftr"))
output_dir = pathlib.Path("/Users/felixachter/source/uni/masterarbeit/half-life-alyx-data-preparation/data/output"
                          "/30_fps-71_users")

# %%

@dataclass
class UserFile:
    file_name: pathlib.Path
    session_idx: int


files_grouped_by_user: Dict[str, List[UserFile]] = dict()

for file in files:
    file_split = file.stem.split("_")
    user = file_split[0]
    existing_files = files_grouped_by_user.get(user, [])
    existing_files.append(UserFile(file_name=file, session_idx=int(file_split[1]) if len(file_split) > 1 else 0))
    files_grouped_by_user[user] = existing_files


# %%
# user, user_files = next(iter(files_grouped_by_user.items()))
for user, user_files in files_grouped_by_user.items():
    user_data = pd.read_feather(user_files[0].file_name)
    if len(user_files) > 1:
        user_data = pd.concat([user_data, pd.read_feather(user_files[1].file_name)])

    output_path = output_dir.joinpath(f"{user}.ftr")
    user_data.reset_index().to_feather(output_path)
# user_data['session_idx']
# if len(user_files) > 1:
#     user_data
# %%
import hashlib

hash_object = hashlib.sha256()
files = sorted(list(output_dir.glob("*.ftr")))
for file_path in files:
    hash_object.update(str(file_path.stem).encode())

hash_value = hash_object.hexdigest()

print(hash_value)
