from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.miners import PairMarginMiner
from torch.distributions import MixtureSameFamily, Categorical, Normal, Independent
from torch.distributions import MultivariateNormal
from tqdm import trange

from src.models.verification_heads import DistanceMatchProbabilityVerificationHead, MatchProbabilityHead
from src.utils.kl_divergence_distance import KLDivDistance
from src.utils.kl_divergence_distance import KLDivDistance as CustomKlDiv
from src.custom_losses.soft_contrastive_loss import loss_soft_con, SoftContrastiveLoss


# from src.utils.kl_distance import KLDivergenceDistance, kl_divergence_variant_2

# %%

def log_probabilities_for_model(match_prob_model: DistanceMatchProbabilityVerificationHead, x_data, y_data):
    unique_subjects = y_data.unique()
    probability_mean_matrix = torch.zeros((len(unique_subjects), len(unique_subjects)), device=x_data.device,
                                          dtype=torch.float32)
    probability_std_matrix = torch.zeros((len(unique_subjects), len(unique_subjects)), device=x_data.device,
                                         dtype=torch.float32)

    for query_subject in unique_subjects:
        for ref_subject in unique_subjects:
            probabilities = match_prob_model(x_data[y_data == query_subject], x_data[y_data == ref_subject])
            probability_mean_matrix[query_subject.int()][ref_subject.int()] = probabilities.mean()
            probability_std_matrix[query_subject.int()][ref_subject.int()] = probabilities.std()

    print(probability_mean_matrix)
    print(probability_std_matrix)

    return probability_mean_matrix, probability_std_matrix


def plot_probs(prob_means, prob_stds):
    mean_numpy_matrix = prob_means.detach().cpu().numpy()
    std_numpy_matrix = prob_stds.detach().cpu().numpy()

    mean_std_matrix = [
        [f"{mean:.4f} \n+- {std:.4f}" for mean, std in zip(mean_row, std_row)]
        for mean_row, std_row in zip(mean_numpy_matrix, std_numpy_matrix)
    ]
    plt.figure(figsize=(16, 16))
    sns.set(font_scale=3)
    sns.set_context("paper", rc={"font.size": 20, "axes.labelsize": 20, "xtick.labelsize": 20,
                                 "ytick.labelsize": 20})  # Custom font sizes
    sns.heatmap(mean_numpy_matrix, annot=mean_std_matrix, fmt='', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Mean +- Std)', fontsize=25)
    plt.show()
# %%

def contrastive_loss(z1, z2, is_match):
    margin = 1.0  # specify the margin value

    distance = (z1 - z2).pow(2).sum(dim=1)  # calculate Euclidean distance

    if is_match:
        loss = distance.mean()
    else:
        loss = F.relu(margin - distance).pow(2).mean()

    return loss


# %%

def get_user_samples(n_samples: int, user_means: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    n_dimensions = user_means.size(1)
    x = torch.zeros((0, n_dimensions), dtype=torch.float32)
    y = torch.zeros((0,), dtype=torch.float32)
    for user_index, user_mean in enumerate(user_means):
        cov = (torch.tensor([0.5 for _ in range(n_dimensions)], dtype=torch.float32)
               * torch.eye(n_dimensions, dtype=torch.float32))
        user_distribution = MultivariateNormal(loc=user_mean,
                                               covariance_matrix=cov)
        user_samples = user_distribution.sample(torch.Size((n_samples,)))
        user_targets = torch.full((n_samples,), user_index, dtype=torch.float32)
        x = torch.concat([x, user_samples])
        y = torch.concat([y, user_targets])
    return x, y


# %%
user_clusters = torch.tensor([
    [0, 0],
    [5, 0],
    [0, 5],
    [5, 5]
], dtype=torch.float32)

x_values, y_values = get_user_samples(10_000, user_clusters)

df = pd.DataFrame(x_values.numpy(), columns=["dim1", "dim2"])
df['target'] = y_values.numpy()

sns.set(style="whitegrid")
fig = plt.figure(figsize=(16, 16))
palette = sns.color_palette("Set1", n_colors=len(df['target'].unique()))
sns.scatterplot(df, x="dim1", y="dim2", hue="target", palette=palette)
plt.title("Scatter Plot of Users")
plt.show()

# %%
lp_distance_func = LpDistance()
# kl_distance_func = KLDivergenceDistance()
lp_loss_func = ContrastiveLoss(distance=lp_distance_func)


# kl_loss_func = ContrastiveLoss(distance=kl_distance_func)


def match_probability(a: torch.Tensor, b: torch.Tensor, point_distances: torch.Tensor):
    return torch.sigmoid(-a * point_distances + b)


def get_match_probs_for(z1: torch.Tensor, z2: torch.Tensor):
    return match_probability(torch.tensor(1), torch.tensor(3), lp_distance_func.pairwise_distance(z1, z2))


# %%
device = torch.device("cuda")


x_values = x_values.to(device=device)
y_values = y_values.to(device=device)
# lp_loss = lp_loss_func.compute_mat(embeddings=x_values[y_values == 0], labels=y_values[y_values == 0],
#                                 ref_emb=x_values[y_values == 1], ref_labels=y_values[y_values == 1])
# distances = lp_distance_func.compute_mat(x_values[y_values == 0], x_values[y_values == 0])
# all_distances_to_first = distances[:, 0]
# print(loss_soft_con(torch.tensor([0., 1., 0.000000001, 0.0000000009]), torch.tensor([1, 1, 0, 0])))

# weight_a = torch.rand(1, requires_grad=True, device=device)
# weight_b = torch.rand(1, requires_grad=True, device=device)

embeddings_zeros = x_values[y_values == 0][:50]
embeddings_ones = x_values[y_values == 1][:50]
first_zeros = y_values[y_values == 0][:50].reshape((-1, 1))
first_ones = y_values[y_values == 1][:50].reshape((-1, 1))

emb = torch.concat([embeddings_zeros, embeddings_ones], dim=0).to(device=device)
gt = torch.concat([first_zeros, first_ones], dim=0).to(device=device)


# %%

miner = PairMarginMiner()
model = DistanceMatchProbabilityVerificationHead()
plot_probs(*log_probabilities_for_model(model.cpu(), x_values.cpu().detach(), y_values.cpu().detach()))
# %%
miner = miner.to(device=device)
model = model.to(device=device)
# %%

optimizer = torch.optim.Adam(model.parameters())

epoch_iterator = trange(1_000)
losses = []
for epoch in epoch_iterator:
    optimizer.zero_grad()
    pair_indices = miner(emb, gt.reshape((-1,)))
    loss_func = SoftContrastiveLoss(model)
    loss = loss_func(emb, gt.reshape((-1,)), pair_indices)
    losses.append(loss.item())
    epoch_iterator.set_postfix_str(f"Loss: {loss.item()}")
    loss.backward()
    optimizer.step()

x_axis_values = list(range(1, len(losses) + 1))
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.lineplot(x=x_axis_values, y=losses)
plt.xlabel("Epochs")
plt.ylabel("Soft Contrastive Loss")  # Replace with your actual metric
plt.title("Learning Curve")
plt.show()

# %%
plot_probs(*log_probabilities_for_model(model.cpu(), x_values.cpu().detach(), y_values.cpu().detach()))
# %%
model = model.to(device=device)
shuffles = torch.randperm(x_values[y_values == 3].size(0)).to(device=device)
positive_predictions = model(x_values[y_values == 3], x_values[y_values == 3][shuffles])
negative_predictions_to_1 = model(x_values[y_values == 3], x_values[y_values == 1])
negative_predictions_to_2 = model(x_values[y_values == 3], x_values[y_values == 2])
negative_predictions_to_0 = model(x_values[y_values == 3], x_values[y_values == 0])

print(positive_predictions.mean(), positive_predictions.std())
print(negative_predictions_to_1.mean(), negative_predictions_to_1.std())
print(negative_predictions_to_2.mean(), negative_predictions_to_2.std())
print(negative_predictions_to_0.mean(), negative_predictions_to_0.std())

# %%
probs = get_match_probs_for(emb[:50], emb[50:])
gt1, gt2 = gt[:50], gt[50:]
current_gt = gt1.eq(gt2).float().reshape((-1,))
log_input = torch.where(current_gt == 1, probs, (1 - probs))
loss = -torch.log(log_input)
print(loss.sum(), loss.mean())
# %%
probs0 = get_match_probs_for(x_values[y_values == 0], x_values[y_values == 0])
probs1 = get_match_probs_for(x_values[y_values == 0], x_values[y_values == 1])
probs2 = get_match_probs_for(x_values[y_values == 0], x_values[y_values == 2])
probs3 = get_match_probs_for(x_values[y_values == 0], x_values[y_values == 3])
print(probs0.mean(), probs0.std())
print(probs1.mean(), probs1.std())
print(probs2.mean(), probs2.std())
print(probs3.mean(), probs3.std())


# %%

class SoftContrastiveLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out


# %%
miner = PairMarginMiner()
anchor_for_positive, positives, anchor_for_negative, negative = miner(emb, gt.reshape((-1,)))

anchors = emb[anchor_for_positive]

# print(probs0.min(), probs0.max(), probs0.mean(), probs0.std())
# print(probs1.min(), probs1.max(), probs1.mean(), probs1.std())
# print(probs2.min(), probs2.max(), probs2.mean(), probs2.std())
# print(probs3.min(), probs3.max(), probs3.mean(), probs3.std())

# user1_distribution = MultivariateNormal(loc=torch.tensor([0., 0.]),
#                                                             covariance_matrix=(torch.tensor([0.5, 0.5]) * torch.eye(2)))
# user1_samples = user1_distribution.sample(torch.Size((10000, )))
# user1_target =
# mean = user1_samples.mean()
# std = user1_samples.std()
# def stochastic_embedding_model(image):
#     # Your implementation here
#     # This function should take an input image and return a distribution over embeddings
#
#     distribution = embedding_distribution
#     return distribution
#
#
# def compute_match_probability(image_1, image_2):
#     K = 8  # Number of Monte-Carlo samples per input image
#
#     embedding_dist_1 = stochastic_embedding_model(image_1)
#     embedding_dist_2 = stochastic_embedding_model(image_2)
#
#     prob_sum = 0
#
#     for k in range(K):
#         z_k_1 = np.random.choice(embedding_dist_samples_from_z_given_x[0], size=embedding_dim)
#         z_k_12 = np.random.choice(embedding_dist_samples_from_z_given_x[0], size=embedding_dim)
#
#         prob_sum += calculate_similarity(z_k_i, z_k_j)
#
#
# match_prob_approximation = (prob_sum / (K **))
#
# return (match_prob_approximation)

# %%


# Construct Gaussian Mixture Model in 1D consisting of 5 equally
# weighted normal distributions
mix = Categorical(torch.ones(5, ))
comp = Normal(torch.randn(5, ), torch.rand(5, ))
gmm = MixtureSameFamily(mix, comp)

# Construct Gaussian Mixture Modle in 2D consisting of 5 equally
# weighted bivariate normal distributions
mix = Categorical(torch.ones(5, ))
comp = Independent(Normal(
    torch.randn(5, 2), torch.rand(5, 2)), 1)
gmm = MixtureSameFamily(mix, comp)

# Construct a batch of 3 Gaussian Mixture Models in 2D each
# consisting of 5 random weighted bivariate normal distributions
mix = Categorical(torch.rand(3, 5))
comp = Independent(Normal(
    torch.randn(3, 5, 2), torch.rand(3, 5, 2)), 1)
gmm = MixtureSameFamily(mix, comp)

results = gmm.sample(torch.Size((1,)))
print(results.size())

# %%

mean = torch.tensor([1.0, 0.0])
cov_matrix = torch.tensor([0.1, 3.0])
normal_dist = Normal(mean, cov_matrix.sqrt())
fig = plt.figure(figsize=(16, 16))
n_samples = 1000
samples = normal_dist.sample((n_samples,))

plt.scatter(samples[:, 0], samples[:, 1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Normal Distributions')
plt.axis('equal')
plt.grid(True)
plt.show()

# %%

user1_means = torch.tensor([[1.0, 0.0], [2.0, 2.0], [0.0, 3.0]])
user1_cov_matrices = torch.tensor([
    [[0.1, 0.], [0., 3.0]],
    [[1., 0.], [0., 1.0]],
    [[1.5, 0.], [0., 0.3]],
])
class1_weights = torch.tensor([1., 1., 1.])

distributions = MultivariateNormal(user1_means, user1_cov_matrices)

# %%

n_dimensions = 2

means = torch.tensor([
    [0., 0.],
    [2., 6.],
    [5., -3.]
], requires_grad=True)
variances = torch.tensor([
    [1., 1.],
    [0.5, 3.],
    [2., 1.]
], requires_grad=True)
cov_matrix = (variances[:, None, :] * torch.eye(n_dimensions).reshape((1, n_dimensions, n_dimensions)))
multi_normal = MultivariateNormal(
    loc=means[:, None, :],
    scale_tril=torch.linalg.cholesky(cov_matrix)[:, None, :])

reference_mean = torch.tensor([
    [0.3, -0.2],
    [2.8, 5.7]
], requires_grad=True)
reference_var = torch.tensor([
    [0.3, 0.5],
    [3., 5.]
], requires_grad=True)
reference_cov_matrix = reference_var[:, None, :] * torch.eye(n_dimensions).reshape((1, n_dimensions, n_dimensions))
references = MultivariateNormal(
    loc=reference_mean[None, :, :],
    scale_tril=torch.linalg.cholesky(reference_cov_matrix)[None, :, :]
)
# %%
custom_kl_div = CustomKlDiv()

q = torch.cat([means, variances], dim=1)
r = torch.cat([reference_mean, reference_var], dim=1)
dist = custom_kl_div(q, r)
print(dist)
# pair_dist = custom_kl_div.pairwise_distance(q, r)
# %%
# references_test = MultivariateNormal(
#     loc=reference_mean,
#     scale_tril=torch.linalg.cholesky(reference_cov_matrix)
# )

# %%
n_samples = 8
multi_samples = multi_normal.sample((n_samples,))[:, :, 0, :].permute(1, 0, 2).reshape(-1, 2).numpy()
# first_user_samples = multi_samples[0].numpy()
# second_user_samples = multi_samples[1].numpy()
# third_user_samples = multi_samples[2].numpy()
class_labels = np.array(["Class 0"] * n_samples + ["Class 1"] * n_samples + ["Class 2"] * n_samples)
distribution_data = pd.DataFrame(multi_samples, columns=['x', 'y'])
distribution_data['target'] = class_labels

reference_data = pd.DataFrame(references.sample((n_samples,))[:, 0, :, :].permute(1, 0, 2).reshape(-1, 2).numpy(),
                              columns=['x', 'y'])
ref_labels = np.array(['Reference 0'] * n_samples + ['Reference 1'] * n_samples)
reference_data['target'] = ref_labels
distribution_data = pd.concat([distribution_data, reference_data], ignore_index=True)
# %%
_, ax = plt.subplots(figsize=(16, 16))
sns.kdeplot(data=distribution_data,
            x='x', y='y', hue='target',
            fill=True, palette=sns.color_palette(),
            levels=10, thresh=.2)
plt.show()
# %%
kl_div = torch.distributions.kl_divergence(multi_normal, references)
print(kl_div)
# %%
flattened_kl_divs = kl_div.flatten()
kl_match = MatchProbabilityHead()
probs = kl_match(flattened_kl_divs)
multi_truth = torch.tensor([0, 1, 2])
ref_truth = torch.tensor([0, 1])
truth = multi_truth[:, None].eq(ref_truth[None, :]).float().flatten()
# truth = torch.tensor([1, 0, 0, 1, 0, 0])
soft_loss = loss_soft_con(probs, truth)
print(probs)
print(truth)
print(soft_loss)
# soft_loss.mean().backward()
# %%


# %%
q_emb = torch.rand((20, 4))
r_emb = torch.rand((10, 4))

dist_func = KLDivDistance()
mat = dist_func(q_emb, r_emb)
pair = dist_func.pairwise_distance(q_emb[:10], r_emb)
# %%
size = torch._C._infer_size(multi_normal._unbroadcasted_scale_tril.shape[:-2],
                            references._unbroadcasted_scale_tril.shape[:-2])
print(size)

# %%
categorical_dist = Categorical(logits=torch.tensor([0, 1, 2]))
cat_samples = categorical_dist.sample((100,))

# %%
cluster = torch.tensor([
    [0.1, -0.1],
    [2., 5.],
    [-3., 3.],
    [4.3, -0.4],
    [-2., -3.]
])
user_values = dict()
n_samples = 1000
n_dimensions = 2

x, y = get_user_samples(n_samples, cluster)
random_variances = torch.rand((n_samples * len(cluster), n_dimensions))
x = torch.concat([x, random_variances], dim=1)
# %%
device = torch.device("cuda")
# %%
x = x.to(device=device)
y = y.to(device=device)
# %%
dist_func = KLDivDistance()

miner = PairMarginMiner(distance=dist_func).to(device=device)

model = DistanceMatchProbabilityVerificationHead(distance=dist_func).to(device=device)
# %%

predictions = model(x[y == 0][:50], x[y == 0][50:100]).to(device=device)

# %%

embeddings_zeros = x[y == 0][:50]
embeddings_ones = x[y == 1][:50]
first_zeros = y[y == 0][:50].reshape((-1, 1))
first_ones = y[y == 1][:50].reshape((-1, 1))

emb = torch.concat([embeddings_zeros, embeddings_ones], dim=0).to(device=device)
gt = torch.concat([first_zeros, first_ones], dim=0).to(device=device)



# %%
plot_probs(*log_probabilities_for_model(model, x, y.reshape((-1,))))
# %%

optimizer = torch.optim.Adam(model.parameters())

epoch_iterator = trange(10_000)
losses = []
for epoch in epoch_iterator:
    optimizer.zero_grad()
    pair_indices = miner(emb, gt.reshape((-1,)))
    loss_func = SoftContrastiveLoss(model)
    loss = loss_func(emb, gt.reshape((-1,)), pair_indices)
    losses.append(loss.item())
    epoch_iterator.set_postfix_str(f"Loss: {loss.item()}")
    loss.backward()
    optimizer.step()

x_axis_values = list(range(1, len(losses) + 1))
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.lineplot(x=x_axis_values, y=losses)
plt.xlabel("Epochs")
plt.ylabel("Soft Contrastive Loss")  # Replace with your actual metric
plt.title("Learning Curve")
plt.show()

# %%
plot_probs(*log_probabilities_for_model(model, x, y.reshape((-1,))))
