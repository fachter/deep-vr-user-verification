import torch
import torch.distributions as dist
from scipy.spatial.distance import mahalanobis
from src.custom_distances.mahalanobis_distance import mahalanobis_distance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# %%
torch.manual_seed(42)

n = 100
d = 2

references = torch.randn((n, d))
query = torch.randn((3, d)) * torch.tensor([[11., -5., 0.3]]).T

mean = torch.mean(references, dim=0)
cov_matrix = torch.cov(references.T)

# %%
sns.scatterplot(x=references[:, 0], y=references[:, 1], color="blue")
sns.scatterplot(x=query[:, 0], y=query[:, 1], hue=[1, 2, 3], palette="hot")
plt.show()
# %%

distribution = dist.MultivariateNormal(mean, cov_matrix)
print("Mahalanobis Distances (torch.distributions):")
print(torch.sqrt(distribution.log_prob(query[0]) * -2.))
print(torch.sqrt(distribution.log_prob(query[1]) * -2.))
print(torch.sqrt(distribution.log_prob(query[2]) * -2.))
print(torch.sqrt(distribution.log_prob(query) * -2.))
# %%

print(mahalanobis(query[0], mean, np.linalg.inv(cov_matrix)))
print(mahalanobis(query[1], mean, np.linalg.inv(cov_matrix)))
print(mahalanobis(query[2], mean, np.linalg.inv(cov_matrix)))


# %%
# def custom_mahalanobis(que, mean_vec, cov):
#     x = que - mean_vec
#     inv_cov_matrix = torch.inverse(cov)
#     left_term = (x @ inv_cov_matrix)
#     full_term = left_term @ x.T
#     diagonal = full_term.diagonal() if full_term.dim() == 2 else full_term
#     return torch.sqrt(diagonal)


print(mahalanobis_distance(query[0], references))
print(mahalanobis_distance(query[1], references))
print(mahalanobis_distance(query[2], references))
print(mahalanobis_distance(query, references))

# %%


# def mahalanobis_distance(q_emb: torch.tensor, mean_emb: torch.tensor, cov: torch.tensor):
#     x = q_emb - mean_emb
#     inv_cov_matrix = torch.inverse(cov)
#     left_term = x @ inv_cov_matrix
#     mahalanobis_dist = torch.sqrt((left_term @ x.T).sum(dim=0))
#     return mahalanobis_dist


def mahalanobis(u: torch.tensor, v: torch.tensor, cov: torch.tensor):
    delta = u - v
    m = torch.dot(delta, torch.matmul(torch.inverse(cov), delta))
    return torch.sqrt(m)


distances = mahalanobis_distance(query, mean, cov_matrix)
print("Mahalanobis Distances (self):")
print(distances)


distances = mahalanobis(query, mean, cov_matrix)
print("Mahalanobis Distances (Scipy):")
print(distances)

# %%

references1 = torch.randn((n, d)) * torch.tensor([0.1, -0.1])
references2 = torch.randn((n, d)) * torch.tensor([0., 0.])