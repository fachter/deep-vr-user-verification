import torch
from time import time
from src.utils.old_kl_distance import KLDivDistance as CustomKLDiv
from src.utils.kl_divergence_distance import KLDivDistance


query_embeddings = torch.rand((200, 192))
ref_embeddings = torch.rand((200, 192))

custom_kl = CustomKLDiv()
kl = KLDivDistance()

start_custom = time()
dist_custom_mat = custom_kl(query_embeddings, ref_embeddings)
dist_custom_pair = custom_kl.pairwise_distance(query_embeddings, ref_embeddings)
end_custom = time()

start_torch = time()
dist_kl_mat = kl(query_embeddings, ref_embeddings)
dist_kl_pair = kl.pairwise_distance(query_embeddings, ref_embeddings)
end_torch = time()

print("TORCH", end_torch - start_torch)
print("CUSTOM", end_custom - start_custom)



