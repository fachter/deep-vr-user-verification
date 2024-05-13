import torch
import numpy as np
import pandas as pd

# %%

columns = [f"col{i}" for i in range(18)]
data = np.random.randn(10_000, 18)
df = pd.DataFrame(data, columns=columns)
users = np.random.randint(0, 10, 10_000)
df['subject'] = users
df_by_user = {user_id: df[df.subject == user_id] for user_id in np.unique(df['subject'])}

# %%

means = np.nanmean(df[columns], axis=0)
stds = np.nanstd(df[columns], axis=0)

# %%

mean_list = []
std_list = []
total_rows = 0
for user_data in df_by_user.values():
    row_count = len(user_data)
    total_rows += row_count
    mean_list.append(np.nanmean(user_data[columns], axis=0) * row_count)
    std_list.append(np.nanstd(user_data[columns], axis=0) * row_count)

mean_list = np.array(mean_list).sum(axis=0) / total_rows
std_list = np.array(std_list).sum(axis=0) / total_rows

# %%

print(np.allclose(means, mean_list))
print(np.allclose(stds, std_list))

