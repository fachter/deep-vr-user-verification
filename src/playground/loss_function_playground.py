import torch
import torch.nn as nn
import pytorch_metric_learning.losses as losses
import pytorch_metric_learning.distances as distances

# %%
# distance = distances.LpDistance(normalize_embeddings=False, p=2)
embeddings = torch.randn((10000, 32))

# dist = distance(embeddings, embeddings)
# print(dist.min(), dist.max())
distance = distances.LpDistance(normalize_embeddings=True, p=2)
normalized_dist = distance(embeddings, embeddings)
# print(dist.min(), dist.max())
print(normalized_dist.min(), normalized_dist.max())

# %%
query = nn.functional.normalize(embeddings[2].reshape((-1, 2)))
ref = nn.functional.normalize(embeddings[3].reshape((-1, 2)))
print(nn.functional.pairwise_distance(
    ((query)),
    ((ref))
))
# %%

normed_embs100 = nn.functional.normalize(embeddings[:100])
normed_embs1000 = nn.functional.normalize(embeddings[:1000])
# %%

import torch

x = torch.tensor([[0.4411, -0.8975]])
y = torch.tensor([[-0.9257, -0.3783]])

# Calculate the L2 (Euclidean) norm of vectors x and y
norm_x = torch.norm(x, p=2)
norm_y = torch.norm(y, p=2)

# Check if the norms are equal to 1
is_x_normalized = torch.isclose(norm_x, torch.tensor(1.000))
is_y_normalized = torch.isclose(norm_y, torch.tensor(1.0))


print("Dist", nn.functional.pairwise_distance(norm_x, norm_y))
# %%
print(torch.all(normed_embs100 == normed_embs1000[:100]))
# %%

arc_face = losses.ArcFaceLoss(12, 128)
print(list(arc_face.parameters())[0].size())
