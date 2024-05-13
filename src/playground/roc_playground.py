import torch
import torchmetrics
from pytorch_metric_learning.distances import LpDistance

from src.custom_distances import LpSimilarity, VerificationCosineSimilarity
from src.utils.embeddings import Embeddings
from src.verification_heads import (VerificationHeadBase, ThresholdVerificationHead, SimilarityVerificationHead,
                                    DistanceMatchProbabilityVerificationHead)
from sklearn.metrics import roc_curve
import numpy as np
from src.custom_metrics.verification_evaluator import VerificationEvaluator
from time import time

# %%

start = time()
query_samples = 40_000
reference_samples = 30_000
dimensions = 1024
print("Matrix size: ", query_samples * reference_samples)
queries = torch.randn((query_samples, dimensions))
references = torch.randn((reference_samples, dimensions))
query_labels = torch.randint(0, 25, (query_samples,))
reference_labels = torch.randint(0, 25, (reference_samples,))

# %%
lp_similarity_head = SimilarityVerificationHead(LpSimilarity())
preds = lp_similarity_head(queries, references)
print("Preds done")
gt = query_labels[:, None].eq(reference_labels[None]).long()
print("GT")
# %%
roc = torchmetrics.classification.BinaryROC()
fpr, tpr, thresholds = roc(preds.flatten(), gt.flatten())
print("ROC")
fnr = 1 - tpr

fnr_fpr = torch.absolute(fnr - fpr)
index_for_eer = torch.argmin(torch.absolute(fnr_fpr))
EER = fpr[index_for_eer]
EER2 = fnr[index_for_eer]
print("EER for similarity head", EER, EER2)
print("EER threshold", thresholds[index_for_eer])
# %%
far = fpr
frr = fnr
threshold_indices = {f"TAR@FAR={far_threshold * 100}%": torch.argmin(torch.abs(far - far_threshold))
                     for far_threshold in [1e-2, 1e-3, 1e-4, 1e-5]}
tpr_at_far = {key: tpr[value] for key, value in threshold_indices.items()}
print(tpr_at_far)

evaluator = VerificationEvaluator(lp_similarity_head)
embeddings = Embeddings(queries, references, query_labels, reference_labels)
scores = evaluator.get_scores(embeddings)
print("Scores")
print(scores)


# %%

def denormalize(value, min_val, max_val, distance_is_inverted=False):
    if not distance_is_inverted:
        value = 1 - value
    return value * (max_val - min_val) + min_val
#
#
# def roc_from_sklearn(ground_truth, predictions):
#     metrics = roc_curve(ground_truth, predictions)
#     return torch.tensor(metrics[0]), torch.tensor(metrics[1]), torch.tensor(metrics[2])
#
#
# def scores(TP, FP, TN, FN):
#     FPR = FP / (FP + TN)
#     FNR = FN / (FN + TP)
#     TPR = TP / (TP + FN)
#     TNR = TN / (TN + FP)
#     return {
#         'FPR': FPR,
#         'FNR': FNR,
#         'TPR': TPR,
#         'TNR': TNR,
#     }
#
# %%
threshold_head = ThresholdVerificationHead(LpDistance(normalize_embeddings=False))
normalized_distances = threshold_head.normalized_to_similarity(queries, references)
max_value, min_value = threshold_head.max_value, threshold_head.min_value

threshold_roc = torchmetrics.classification.BinaryROC()
fpr, tpr, thresholds = threshold_roc(normalized_distances.flatten(), gt.flatten())
eer_threshold = thresholds[np.nanargmin(np.abs((1 - tpr) - fpr))]
eer = (1 - tpr[np.nanargmin(np.abs((1 - tpr) - fpr))])

print(f"Estimated EER: {eer} at threshold: {eer_threshold}")
threshold_indices = {f"TAR@FAR={far_threshold * 100}%": torch.argmin(torch.abs(far - far_threshold))
                     for far_threshold in [1e-2, 1e-3, 1e-4, 1e-5]}
tpr_at_far = {key: tpr[value] for key, value in threshold_indices.items()}
print(tpr_at_far)

denormalized_threshold = denormalize(eer_threshold, min_value, max_value, False)
matches = threshold_head(queries, references, denormalized_threshold)

threshold_roc = torchmetrics.classification.BinaryROC()
fpr, tpr, new_thresholds = threshold_roc(matches.flatten(), gt.flatten())
fnr = 1 - tpr
best_index = np.nanargmin(np.abs((1 - tpr) - fpr))
eer_threshold = new_thresholds[best_index]
eer = (1 - tpr[best_index])

print(f"Estimated EER: {eer} at threshold (denormalized): {denormalized_threshold}")
denormalized_scores = dict()
for threshold in threshold_indices.values():
    d_threshold = denormalize(thresholds[threshold], min_value, max_value, False)
    matches = threshold_head(queries, references, d_threshold)
    threshold_roc = torchmetrics.classification.BinaryROC()
    fpr, tpr, new_thresholds = threshold_roc(matches.flatten(), gt.flatten())
    denormalized_scores[str(d_threshold.item())] = tpr[1]
print(denormalized_scores)

threshold_evaluator = VerificationEvaluator(threshold_head)
threshold_scores = threshold_evaluator.get_scores(embeddings)
print(threshold_scores)
#
#
# scorer = torchmetrics.classification.BinaryStatScores()
# tp, fp, tn, fn, sup = scorer(matches.flatten(), gt.flatten())
# print(scores(tp, fp, tn, fn))
#
#
# # %%
#
# threshold_head = ThresholdVerificationHead(LpDistance(normalize_embeddings=False))
# normalized_distances, max_value, min_value = threshold_head.normalized_to_similarity(queries, references)
# threshold_roc = torchmetrics.classification.BinaryROC()
# fpr, tpr, thresholds = threshold_roc(normalized_distances.flatten(), gt.flatten())
# fnr = 1 - tpr
# fnr_fpr = torch.absolute(fnr - fpr)
# index_for_eer = torch.argmin(fnr_fpr)
# EER = fpr[index_for_eer]
# EER2 = fnr[index_for_eer]
# print("EER for threshold head with normalized distances", EER, EER2, tpr[index_for_eer])
#
# # %%
#
# threshold_head = ThresholdVerificationHead(LpDistance(normalize_embeddings=False))
# normalized_distances, max_value, min_value = threshold_head.normalized_to_similarity(queries, references)
# threshold_roc = torchmetrics.classification.BinaryROC()
# # fpr, tpr, thresholds = threshold_roc(normalized_distances.flatten(), gt.flatten())
# fpr, tpr, thresholds = roc_from_sklearn(gt.flatten(), normalized_distances.flatten())
# fnr = 1 - tpr
# fnr_fpr = torch.absolute(fnr - fpr)
# index_for_eer = torch.argmin(fnr_fpr)
# EER = fpr[index_for_eer]
# EER2 = fnr[index_for_eer]
# print("EER / TPR from sklearn for threshold head with normalized distances", EER, EER2, tpr[index_for_eer])
#
# # %%
#
# used_threshold = thresholds[index_for_eer]
# denormalized_threshold = denormalize(used_threshold, min_value, max_value)
# print("denormalized threshold", denormalized_threshold)
# # %%
# # print((torch.tensor(4.18) - min_value) / (max_value - min_value))
# # print(denormalize((torch.tensor(4.18) - min_value) / (max_value - min_value), min_value, max_value))
# # %%
#
# matches = threshold_head(queries, references, denormalized_threshold)
# scorer = torchmetrics.classification.BinaryStatScores()
# tp, fp, tn, fn, sup = scorer(matches.flatten(), gt.flatten())
#
# fpr = (fp / (fp + tn)).item()
# fnr = (fn / (fn + tp)).item()
# tpr = (tp / (tp + fn)).item()
# # print("FNR / FPR / TPR for denormalized threshold", fnr, fpr, tpr, fpr + tpr)
# print("FNR denormalized threshold", fnr)
# print("FPR denormalized threshold", fpr)
# print("TPR denormalized threshold", tpr)
#
# # %%
#
# fpr, tpr, thresholds = roc_curve(gt.flatten(), matches.flatten())
# fnr = 1 - tpr
# print("FPR sklearn", fpr[1])
# print("TPR sklearn", tpr[1])
# print("FNR sklearn", fnr[1])
#
#
# # %&
#
# values = scores(tp, fp, tn, fn)
# print(values)
# print(1 - values['TPR'], values['FNR'])

end = time()
print("Final time", end - start)
# %%

# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)
