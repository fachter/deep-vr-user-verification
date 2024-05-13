import torch
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity
from src.models.verification_heads import _mahalanobis
from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
from src.custom_distances import KLDivDistance, VerificationCosineSimilarity, LpSimilarity

# %%

references = torch.FloatTensor([
    [2, 5],
    [2.2, 4.8],
    [1.8, 4.7]
])

queries = torch.FloatTensor([
    [2.1, 4.9],
    [-3.1, 12.5],
    [0, 0],
    [100.5, -153],
    [-999999999., -125892628]
])

euclidean_normalized = LpDistance(normalize_embeddings=True, p=2)
euclidean_non_normalized = LpDistance(normalize_embeddings=False, p=2)
manhattan_normalized = LpDistance(normalize_embeddings=True, p=1)
manhattan_non_normalized = LpDistance(normalize_embeddings=False, p=1)
p10_normalized = LpDistance(normalize_embeddings=True, p=10)
p10_non_normalized = LpDistance(normalize_embeddings=False, p=10)
cosine_similarity = CosineSimilarity()

verification_cosine_similarity = VerificationCosineSimilarity()
euclidean_similarity = LpSimilarity()
manhattan_similarity = LpSimilarity(p=1)


def print_distances(func, text):
    print(text)
    dist = func(queries, references)
    print("Min", dist.min())
    print("Max", dist.max())
    print(dist)
    print()


print_distances(euclidean_normalized, "p=2; normalized")
print_distances(euclidean_non_normalized, "p=2; non normalized")
print_distances(manhattan_normalized, "p=1; normalized")
print_distances(manhattan_non_normalized, "p=1; non normalized")
print_distances(p10_normalized, "p=10; normalized")
print_distances(p10_non_normalized, "p=10; non normalized")
print_distances(cosine_similarity, "cosine similarity")

print_distances(verification_cosine_similarity, "verification cosine similarity")
print_distances(manhattan_similarity, "p=1; similarity")
print_distances(euclidean_similarity, "p=2; similarity")

# %%
# queries_normalized = torch.nn.functional.normalize(queries, p=2, dim=1)
# references_normalized = torch.nn.functional.normalize(references, p=2, dim=1)
#
# dist_mahalanobis = _mahalanobis(queries, references)
# dist_norm_mahalanobis = _mahalanobis(queries_normalized, references_normalized)
#
# print("Non normalized")
# print("Min", dist_mahalanobis.min())
# print("Max", dist_mahalanobis.max())
# print(dist_mahalanobis)
# print()
# print("normalized")
# print("Min", dist_norm_mahalanobis.min())
# print("Max", dist_norm_mahalanobis.max())
# print(dist_norm_mahalanobis)
# print()
#
# similarity_scores = torch.exp(-dist_mahalanobis)
# similarity_norm_scores = torch.exp(-dist_norm_mahalanobis)
#
# normalized_similarity_scores = 1 / (1 + similarity_scores)
# normalized_similarity_norm_scores = 1 / (1 + similarity_norm_scores)
#
# print("Scores")
# print(similarity_scores.min())
# print(similarity_scores.max())
# print(similarity_scores)
# print()
# print("Scores for normed distances")
# print(similarity_norm_scores.min())
# print(similarity_norm_scores.max())
# print(similarity_norm_scores)
# print()
# print("Normed Scores")
# print(normalized_similarity_scores.min())
# print(normalized_similarity_scores.max())
# print(normalized_similarity_scores)
# print()
# print("Normed Scores for normed distances")
# print(normalized_similarity_norm_scores.min())
# print(normalized_similarity_norm_scores.max())
# print(normalized_similarity_norm_scores)
#
# %%
euclidean_similarity_match_finder = MatchFinder(euclidean_similarity)
cosine_similarity_match_finder = MatchFinder(verification_cosine_similarity)

euclidean_matches = euclidean_similarity_match_finder.get_matching_pairs(queries, references, threshold=0.7)
cosine_matches = cosine_similarity_match_finder.get_matching_pairs(queries, references, threshold=0.9)

print("\nEuclidean matches")
print(torch.tensor(euclidean_matches).float())
print("\nCosine matches")
print(torch.tensor(cosine_matches).float())

