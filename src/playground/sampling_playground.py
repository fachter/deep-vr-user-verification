import torch
import numpy as np
import os
from src.custom_distances import KLDivDistance
# %%
dimensions = 4
cut = dimensions // 2
x_query = torch.rand((1, dimensions))
x_ref = torch.rand((1, dimensions))

mean_query, sigma_query = x_query[:, :cut], x_query[:, cut:]
mean_ref, sigma_ref = x_ref[:, :cut], x_ref[:, cut:]

k = 8

# %%
epsilon_query = torch.randn((k,) + mean_query.shape)
epsilon_query = epsilon_query.unsqueeze(-2)
epsilon_ref = torch.randn((k,) + mean_ref.shape)
epsilon_ref = epsilon_ref.unsqueeze(-2)
# %%
sigma_query_matrix = torch.diag_embed(sigma_query)
sigma_ref_matrix = torch.diag_embed(sigma_ref)

samples_query = (mean_query.unsqueeze(1) + epsilon_query @ sigma_query_matrix.unsqueeze(0)).squeeze(2)
samples_ref = (mean_ref.unsqueeze(1) + epsilon_ref @ sigma_ref_matrix.unsqueeze(0)).squeeze(2)

# %%

query = torch.concat([samples_query.mean(dim=0), samples_query.var(dim=0)], dim=1)
ref = torch.concat([samples_ref.mean(dim=0), samples_ref.var(dim=0)], dim=1)

dist_func = KLDivDistance()
dist_mat = dist_func(query, ref)

print(dist_mat)

# %%
sigma_query.reshape((-1, cut)), epsilon_query.reshape((-1, cut)) * sigma_query.reshape((-1, cut))
# %%

(k, ) + mean_query.size()
