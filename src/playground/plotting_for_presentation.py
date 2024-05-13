import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
fig, ax = plt.subplots(1, 1)
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 2)

user1_data = np.array([
    [0.9, 1.2],
    [1.1, 0.8],
    [1.0, 1.0],
    [0.8, 0.9],
    [0.7, 1.0],
])
user2_data = np.array([
    [-0.9, -1.0],
    [-0.7, -1.1],
    [-1.0, -1.2],
    [-0.8, -0.9],
    [-1.1, -1.0],
])
user3_data = np.array([
    [2.5, 0.7],
    [2.7, 0.8],
    [2.6, 0.6],
    [2.7, 0.7],
    [2.6, 0.8],
])
total_data = np.concatenate([
    user1_data,
    # user2_data,
    # user3_data
], axis=0)
user_target = np.array(
    [0] * 5
    # + [1] * 5
    # + [2] * 5
)
df = pd.DataFrame(data=total_data, columns=["x1", "x2"])
df['target'] = user_target
df.loc[len(df)] = [0, 1.5, "?"]
palette = sns.color_palette("rocket")
# palette = [palette[0], palette[2], palette[3], palette[5]]
# palette = [palette[0], palette[2], palette[3]]
palette = [palette[0], palette[5]]
sns.scatterplot(data=df, x="x1", y="x2", hue="target", ax=ax, palette=palette)

plt.legend(loc="upper left")
plt.show()

# %%

# %%

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# Create data for class 1 distributions
class1_distributions = [
    {"mean": [2, 3], "cov": [[1, 0.5], [0.5, 1]], "label": "1"},
    {"mean": [6, 7], "cov": [[1, 0.3], [0.3, 1]], "label": "1"},
    {"mean": [0, 8], "cov": [[0.8, 0.2], [0.2, 0.8]], "label": "1"},
]

# Create data for class 2 distributions
class2_distributions = [
    {"mean": [-10, -8], "cov": [[1, 0.5], [0.5, 1]], "label": "2"},
    {"mean": [-12, -2], "cov": [[1, 0.3], [0.3, 1]], "label": "2"},
    {"mean": [-16, -7], "cov": [[0.8, 0.2], [0.2, 0.8]], "label": "2"},
]

# Create a figure and axes
fig, ax = plt.subplots()

# Plot each distribution in class 1
for distribution in class1_distributions:
    data = np.random.multivariate_normal(distribution["mean"], distribution["cov"], 1000)
    sns.kdeplot(x=data[:, 0], y=data[:, 1], fill=True, cmap="Blues", ax=ax, label=distribution["label"])

# Plot each distribution in class 2
for distribution in class2_distributions:
    data = np.random.multivariate_normal(distribution["mean"], distribution["cov"], 1000)
    sns.kdeplot(x=data[:, 0], y=data[:, 1], fill=True, cmap="Oranges", ax=ax, label=distribution["label"])

# Set axis labels
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
#
# # Set plot title
# plt.title("2D Distributions for Class 1 and Class 2")

# Manually create a legend with unique labels
handles, labels = ax.get_legend_handles_labels()
unique_labels = list(set(labels))
handles = [
    Line2D([0], [0], lw=4, label="1", color="Orange"),
    Line2D([0], [0], lw=4, label="0"),
           ]
ax.legend(handles, sorted(unique_labels), loc="upper left")
ax.set_xlim(-25, 15)
ax.set_ylim(-15, 15)
# Show the plot
plt.show()

