import torch
import torchmetrics
from time import time
from torchmetrics.classification import BinaryStatScores
import numpy as np
from pytorch_metric_learning.distances import LpDistance

from src.custom_metrics.verification_accuracy_calculator import _get_random_negative_values
from src.models.verification_heads import (MahalanobisThresholdVerificationHead,
                                           MatchProbabilityHead,
                                           DistanceMatchProbabilityVerificationHead,
                                           NClosestThresholdVerificationHead, _mahalanobis)
from src.utils.embeddings import Embeddings

# %%

device = torch.device("cuda")

n_dim = 192
query_embeddings = torch.randn((5500, n_dim), dtype=torch.float32, device=device)
reference_embeddings = torch.randn((2500, n_dim), dtype=torch.float32, device=device)

query_labels = torch.randint(0, 11, (5500,), device=device)
reference_labels = torch.randint(0, 11, (2500,), device=device)

dist_maha = _mahalanobis(query_embeddings, reference_embeddings)
print(dist_maha.min(), dist_maha.max(), dist_maha.mean(), dist_maha.std())

# %%

embeddings = Embeddings(
    query=query_embeddings,
    reference=reference_embeddings,
    query_labels=query_labels,
    reference_labels=reference_labels
)

unique_users = embeddings.query_labels.unique()
negative_users = {
    user_id.int(): _get_random_negative_values(
        unique_users, user_id.int().item())[:3]
    for user_id in unique_users
}

verification_head = MahalanobisThresholdVerificationHead()


# verification_head = NClosestThresholdVerificationHead()
# verification_head = DistanceMatchProbabilityVerificationHead()


def get_matches_and_truth(embs: Embeddings, reference_user, query_user, threshold=None):
    ref = embs.reference[embs.reference_labels == reference_user]
    query_label_mask = torch.isin(embs.query_labels, query_user)
    que = embs.query[query_label_mask]
    matches = verification_head(que, ref, threshold)
    truth = verification_head.get_truth(embs, query_user, reference_user)

    return matches, truth


def get_far_and_frr_for_threshold(embs: Embeddings, threshold: float):
    stat_threshold = threshold if verification_head.match_probability_head else 0.5
    binary_stat_scores = BinaryStatScores(threshold=stat_threshold, multidim_average="global").to(embs.query.device)
    scores_per_user = {(ref_user.item(), ref_user.item()): binary_stat_scores(
        *get_matches_and_truth(embs, ref_user, ref_user, threshold))
        for ref_user in negative_users.keys()
    }
    for ref_user, query_users in negative_users.items():
        for query_user in query_users:
            matches, truth = get_matches_and_truth(embs, ref_user, query_user, threshold)
            scores_per_user[(ref_user.item(), query_user.item())] = binary_stat_scores(matches, truth)
    total_average = torch.cat(
        [score_per_user[None] for score_per_user in scores_per_user.values()], dim=0).sum(dim=0)
    (tp, fp, tn, fn, sup) = total_average
    far = torch.nan_to_num(fp / (fp + tn), nan=1.0)
    frr = torch.nan_to_num(fn / (fn + tp), nan=1.0)
    return far, frr


def get_eer(embs: Embeddings, min_threshold: float, max_threshold: float):
    threshold = (min_threshold + max_threshold) / 2.
    far, frr = get_far_and_frr_for_threshold(embs, threshold)
    if torch.isclose(far, frr, 0.001) or abs(max_threshold - min_threshold) < 1e-10:
        return threshold, frr
    far_frr = far < frr
    if verification_head.match_probability_head:
        far_frr = not far_frr
    if far_frr:
        return get_eer(embeddings, threshold, max_threshold)
    return get_eer(embeddings, min_threshold, threshold)


# %%
print(f"Head: {verification_head.__class__.__name__}")

if verification_head.match_probability_head:
    start_1 = time()
    result = get_eer(embeddings, 0., 1.)
    end_1 = time()
    print(f"1: {result} in {end_1 - start_1}")
else:
    start_100 = time()
    result = get_eer(embeddings, 0., 100.)
    end_100 = time()
    print(f"100: {result} in {end_100 - start_100}")

    start_1000 = time()
    result = get_eer(embeddings, 0., 1000.)
    end_1000 = time()
    print(f"1000: {result} in {end_1000 - start_1000}")

    start_5000 = time()
    result = get_eer(embeddings, 0., 5000.)
    end_5000 = time()
    print(f"5000: {result} in {end_5000 - start_5000}")

    start_10000 = time()
    result = get_eer(embeddings, 0., 10000.)
    end_10000 = time()
    print(f"10000: {result} in {end_10000 - start_10000}")
