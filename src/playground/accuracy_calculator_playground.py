import time
from typing import Dict, Tuple

import torch
import torchmetrics
import pytorch_metric_learning.utils.accuracy_calculator as accuracy_utils
from pytorch_metric_learning.distances import LpDistance

from src.custom_metrics.verification_accuracy_calculator import VerificationScoreCalculator
from src.lightning_modules.verification_embedding_module import VerificationEmbeddingModule
from src.models.cnn_model import CNNModel
from src.utils.kl_divergence_distance import KLDivDistance
from src.models.verification_heads import DistanceMatchProbabilityVerificationHead

# %%
device = torch.device("cuda")
pair_match_probability_head = DistanceMatchProbabilityVerificationHead()
verification_evaluator = VerificationScoreCalculator(
    thresholds={.5, .7},
    return_per_class=True,
    verification_head=pair_match_probability_head,
    number_of_negative_users=3
).to(device=device)
score = verification_evaluator.score_functions["50"](torch.randn((100,)).to(device=device),
                                                     torch.randint(1, (100,)).to(device=device))
auth_scores = verification_evaluator.get_scores(torch.randn((500, 192)).to(device=device), torch.randint(0, 11, (500,)).to(device=device),
                                                torch.randn((500, 192)).to(device=device), torch.randint(0, 11, (500,)).to(device=device), )
# %%

# accuracy_utils.precision_at_k(1)
n_samples = 1_000_000
# n_samples = 10_000
device = torch.device("cpu")
index_to_skip = torch.randint(0, 3, (1,))
query_mask = torch.arange(n_samples) % 4 != index_to_skip

query = torch.rand((n_samples, 192), device=device)[::5]
query_labels = torch.randint(0, 11, (n_samples,), device=device)[::5]

references = torch.rand((n_samples, 192), device=device)[::150]
reference_labels = torch.randint(0, 11, (n_samples,), device=device)[::150]

# %%
distance = LpDistance()
distances = distance(query, references)

# %%
match_probability = DistanceMatchProbabilityVerificationHead()
evaluator = VerificationScoreCalculator(
    thresholds=[0.5, 0.7, .9, .95],
    verification_head=match_probability,
    return_per_class=True,
    device=device
)
# %%
start = time.time()
prob_cpu = match_probability(references.detach().cpu(), query.detach().cpu())
end = time.time()
print("CPU", end - start)
# %%
start = time.time()
iterations = 20
step_size = query.size(0) // iterations
match = torch.empty((references.size(0), 0))
for index in range(iterations):
    shorter_query = query[index * step_size:(index + 1) * step_size]
    match = torch.cat([match, match_probability(references, shorter_query).detach().cpu()], dim=1)
    # probs = match_probability(references, query)
    print(match.size())
end = time.time()
print("CUDA", end - start)
# %%
# evaluator._perform_binary_stat_score_calculation(query, references, query_labels, reference_labels)
# evaluator._fill_scores()
# print(evaluator.total_score_per_threshold)
# print(evaluator.score_per_user_and_threshold)
# %%
# result_first_half = evaluator.get_scores(query[query_labels < 6], query_labels[query_labels < 6],
#                                          references[reference_labels < 6], reference_labels[reference_labels < 6])
# result_second_half = evaluator.get_scores(query[(query_labels >= 6) & (query_labels < 12)],
#                                           query_labels[(query_labels >= 6) & (query_labels < 12)],
#                                           references[(reference_labels >= 6) & (reference_labels < 12)],
#                                           reference_labels[(reference_labels >= 6) & (reference_labels < 12)])
#
# print(result_first_half)
# print(result_second_half)
# %%
result1 = evaluator.get_scores(query, query_labels, references, reference_labels)

print(result1.keys())
print(result1['50'])
# %%
result2 = evaluator.get_scores(query, query_labels, references, reference_labels)

print(result2.keys())
print(result2['50'])
# %%

val = next(iter(zip(result1.values(), result2.values())))

# %%
# scores_50 = torchmetrics.classification.BinaryStatScores(threshold=0.5, multidim_average="global")
# scores_70 = torchmetrics.classification.BinaryStatScores(threshold=0.7, multidim_average="global")
# scores_90 = torchmetrics.classification.BinaryStatScores(threshold=0.9, multidim_average="global")
# scores_95 = torchmetrics.classification.BinaryStatScores(threshold=0.95, multidim_average="global")
# scores = {
#     "50": scores_50,
#     "70": scores_70,
#     "90": scores_90,
#     "95": scores_95
# }
#
#
# # %%
# def get_random_negative_values(unique_user_length: int, number_to_exclude: int):
#     random_values = torch.randperm(unique_user_length)
#     return random_values[random_values != number_to_exclude]
#
#
# n_neg_users = 3
# unique_users = query_labels.unique()
# negative_users = {
#     user_id.int(): get_random_negative_values(len(unique_users), user_id.int().item())[:n_neg_users]
#     for user_id in unique_users
# }
#
# # %%
# kl_div_distance = KLDivDistance()
# match_probability = PairMatchProbabilityHead(distance=kl_div_distance)
#
# # %%
#
# user_score_values: Dict[Tuple[int, int], torch.tensor] = {}
#
#
# def fill_user_scores(user_scores: Dict[Tuple[int, int], torch.tensor],
#                      ref_user: torch.tensor, query_user: torch.tensor):
#     ref = references[reference_labels == ref_user]
#     que = query[query_labels == query_user]
#     query_user_index = query_user.int().item()
#     ref_user_index = ref_user.int().item()
#     matches = match_probability(ref, que)
#     if ref_user == query_user:
#         truth = torch.ones(matches.size())
#     else:
#         truth = torch.zeros(matches.size())
#
#     score_results = {score_key: score(matches, truth) for score_key, score in scores.items()}
#     user_scores[(query_user_index, ref_user_index)] = score_results
#
#
# for user in unique_users:
#     fill_user_scores(user_score_values, user, user)
#
# for ref_user, neg_users in negative_users.items():
#     for neg_user in neg_users:
#         fill_user_scores(user_score_values, ref_user, neg_user)
#
# # %%
#
#
# score_per_user = {}
# combined_scores = {}
#
# for key, user_score in user_score_values.items():
#     for user_score_key, user_score_value in user_score.items():
#         if user_score_key in combined_scores.keys():
#             combined_scores[user_score_key] += user_score_value
#         else:
#             combined_scores[user_score_key] = user_score_value.clone()
#         score_per_user_key = (key[0], user_score_key)
#         if score_per_user_key in score_per_user.keys():
#             score_per_user[score_per_user_key] += user_score_value
#         else:
#             score_per_user[score_per_user_key] = user_score_value.clone()
#
# # %%
#
# tp_50, fp_50, tn_50, fn_50, sup_50 = combined_scores['50']
#
#
# # %%
#
# def get_scores_for_key(score_dict, score_key):
#     tp, fp, tn, fn, sup = score_dict[score_key]
#     false_acceptance_rate = fp / (fp + tn)
#     false_rejection_rate = fn / (fn + tp)
#
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     return false_acceptance_rate, false_rejection_rate, precision, recall
#
#
# # %%
# for score_threshold in ['50', '70', '90', '95']:
#     print(score_threshold, get_scores_for_key(combined_scores, score_threshold))
#
# # %%
# print()
# for user_index in range(10):
#     print(user_index, get_scores_for_key(score_per_user, (user_index, '50')))
