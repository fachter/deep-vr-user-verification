import unittest

import torch
import torch.nn as nn
from pytorch_metric_learning.distances import LpDistance

from src.models.verification_heads import (DistanceMatchProbabilityVerificationHead,
                                           MahalanobisThresholdVerificationHead,
                                           NClosestThresholdVerificationHead,
                                           MatchProbabilityHead)
# MahalanobisMatchProbabilityVerificationHead,

from unittest import TestCase

from src.utils.embeddings import Embeddings


class VerificationHeadsTestCase(TestCase):
    def setUp(self) -> None:
        self.references = torch.tensor([
            [-0.1, 0.1],
            [0., 0.05],
            [0.1, -0.1]
        ])
        self.query = torch.tensor([
            [0.0, 0.0],
            [3., 3.],
            [-5., 0.],
            [0.0, 0.0]
        ])
        self._match_prob_head = MatchProbabilityHead()
        self._match_prob_head.a = nn.Parameter(torch.tensor(1.))
        self._match_prob_head.b = nn.Parameter(torch.tensor(2.))

    def test_match_prob_with_lp_distance(self):
        expected = torch.sigmoid(torch.tensor([
            [1.8586, 1.9500, 1.8586],
            [-2.2450, -2.2074, -2.2450],
            [-2.9010, -3.0002, -3.1010],
            [1.8586, 1.9500, 1.8586]
        ]))
        head = DistanceMatchProbabilityVerificationHead(self._match_prob_head,
                                                        distance=LpDistance(normalize_embeddings=False))

        actual = head(self.query, self.references)

        self.assertTrue(torch.allclose(actual, expected, 1e-04))

    def test_n_closest_head_n_is_1(self):
        expected = torch.tensor([
            1.,
            0.,
            0.,
            1.
        ])
        head = NClosestThresholdVerificationHead(torch.tensor(0.1), n=1)

        actual = head(self.query, self.references)

        self.assertTrue(torch.allclose(actual, expected, 1e-04))

    def test_n_closest_head_n_is_3(self):
        # distance = torch.tensor([
        #     [0.1414, 0.0500, 0.1414], # => 3 / 3
        #     [4.2450, 4.2074, 4.2450], # => 1 / 3
        #     [4.9010, 5.0002, 5.1010]  # => 0 / 3
        # ])
        expected = torch.tensor([
            1.,
            1. / 3.,
            0.,
            1.
        ])
        head = NClosestThresholdVerificationHead(torch.tensor(4.21), n=3)

        actual = head(self.query, self.references)

        self.assertTrue(torch.allclose(actual, expected, 1e-04))

    @unittest.skip
    def test_mahalanobis_match_prob_head(self):
        # torch.tensor([0.5773, 209.4284, 180.8322])
        expected = torch.sigmoid(
            torch.tensor([0.5773, 209.4284, 180.8322, 0.5773]) * torch.tensor(-1.) + torch.tensor(2.))
        head = MahalanobisMatchProbabilityVerificationHead(self._match_prob_head)

        actual = head(self.query, self.references)

        self.assertTrue(torch.allclose(actual, expected, 1e-04))

    def test_mahalanobis_threshold_head_small_threshold(self):
        # torch.tensor([0.5773, 209.4284, 180.8322])
        expected = torch.tensor([1., 0., 0., 1.])
        head = MahalanobisThresholdVerificationHead(threshold=0.8)

        actual = head(self.query, self.references)

        self.assertTrue(torch.allclose(actual, expected, 1e-04))

    def test_mahalanobis_threshold_head_big_threshold(self):
        # torch.tensor([0.5773, 209.4284, 180.8322])
        expected = torch.tensor([1., 0., 1., 1.])
        head = MahalanobisThresholdVerificationHead(threshold=201.3)

        actual = head(self.query, self.references)

        self.assertTrue(torch.allclose(actual, expected, 1e-04))

    def test_mahalanobis_get_truth(self):
        head = MahalanobisThresholdVerificationHead(torch.tensor(1.3))
        que_labels = torch.tensor([1, 2, 1, 2, 1])
        ref_labels = torch.tensor([1, 1, 0, 0])
        embeddings = Embeddings(
            query=torch.randn((len(ref_labels), 2)),
            reference=torch.randn((len(ref_labels), 2)),
            query_labels=que_labels,
            reference_labels=ref_labels
        )
        expected_12 = torch.zeros((2,))
        expected_01 = torch.zeros((3,))
        expected_11 = torch.ones((3,))

        actual_12 = head.get_truth(embeddings, 2, 1)
        actual_01 = head.get_truth(embeddings, 1, 0)
        actual_11 = head.get_truth(embeddings, 1, 1)

        self.assertTrue(torch.all(torch.eq(expected_12, actual_12)))
        self.assertTrue(torch.all(torch.eq(expected_01, actual_01)))
        self.assertTrue(torch.all(torch.eq(expected_11, actual_11)))

    @unittest.skip
    def test_mahalanobis_match_prob_get_truth(self):
        head = MahalanobisMatchProbabilityVerificationHead(self._match_prob_head)
        que_labels = torch.tensor([1, 2, 1, 2, 1])
        ref_labels = torch.tensor([1, 1, 0, 0])
        embeddings = Embeddings(
            query=torch.randn((len(ref_labels), 2)),
            reference=torch.randn((len(ref_labels), 2)),
            query_labels=que_labels,
            reference_labels=ref_labels
        )
        expected_12 = torch.zeros((2,))
        expected_01 = torch.zeros((3,))
        expected_11 = torch.ones((3,))

        actual_12 = head.get_truth(embeddings, 2, 1)
        actual_01 = head.get_truth(embeddings, 1, 0)
        actual_11 = head.get_truth(embeddings, 1, 1)

        self.assertTrue(torch.all(torch.eq(expected_12, actual_12)))
        self.assertTrue(torch.all(torch.eq(expected_01, actual_01)))
        self.assertTrue(torch.all(torch.eq(expected_11, actual_11)))

    def test_n_closest_get_truth(self):
        head = NClosestThresholdVerificationHead(torch.tensor(1.2))
        que_labels = torch.tensor([1, 2, 1, 2, 1])
        ref_labels = torch.tensor([1, 1, 0, 0])
        embeddings = Embeddings(
            query=torch.randn((len(ref_labels), 2)),
            reference=torch.randn((len(ref_labels), 2)),
            query_labels=que_labels,
            reference_labels=ref_labels
        )
        expected_12 = torch.zeros((2,))
        expected_01 = torch.zeros((3,))
        expected_11 = torch.ones((3,))

        actual_12 = head.get_truth(embeddings, 2, 1)
        actual_01 = head.get_truth(embeddings, 1, 0)
        actual_11 = head.get_truth(embeddings, 1, 1)

        self.assertTrue(torch.all(torch.eq(expected_12, actual_12)))
        self.assertTrue(torch.all(torch.eq(expected_01, actual_01)))
        self.assertTrue(torch.all(torch.eq(expected_11, actual_11)))

    def test_mat_prob_get_truth_single_que(self):
        head = DistanceMatchProbabilityVerificationHead(self._match_prob_head)
        que_labels = torch.tensor([1, 2, 1, 2, 1, 2])
        ref_labels = torch.tensor([1, 1, 0, 0])
        embeddings = Embeddings(
            query=torch.randn((len(ref_labels), 2)),
            reference=torch.randn((len(ref_labels), 2)),
            query_labels=que_labels,
            reference_labels=ref_labels
        )
        expected_12 = torch.zeros((3, 2))
        expected_01 = torch.zeros((3, 2))
        expected_11 = torch.ones((3, 2))

        actual_12 = head.get_truth(embeddings, 2, 1)
        actual_01 = head.get_truth(embeddings, 1, 0)
        actual_11 = head.get_truth(embeddings, 1, 1)

        self.assertTrue(torch.all(torch.eq(expected_12, actual_12)))
        self.assertTrue(torch.all(torch.eq(expected_01, actual_01)))
        self.assertTrue(torch.all(torch.eq(expected_11, actual_11)))

    def test_check_same_dimensions_for_match_and_truth_for_every_head(self):
        que_labels = torch.tensor([1, 2, 1, 2, 1, 2, 1, 5, 2])
        ref_labels = torch.tensor([1, 1, 0, 0, 0, 2, 1])
        embeddings = Embeddings(
            query=torch.randn((len(que_labels), 2)),
            reference=torch.randn((len(ref_labels), 2)),
            query_labels=que_labels,
            reference_labels=ref_labels
        )
        heads = [
            DistanceMatchProbabilityVerificationHead(self._match_prob_head),
            # MahalanobisMatchProbabilityVerificationHead(self._match_prob_head),
            MahalanobisThresholdVerificationHead(torch.tensor(5.3)),
            NClosestThresholdVerificationHead(torch.tensor(5.3)),
        ]

        for head in heads:
            for ref_label in [0, 1]:
                for query_label in [1, 2]:
                    matches = head(embeddings.query[embeddings.query_labels == query_label],
                                   embeddings.reference[embeddings.reference_labels == ref_label])
                    truth = head.get_truth(embeddings, query_label, ref_label)
                    self.assertEqual(matches.size(), truth.size(), msg=f"Failed for {head.__class__.__name__}")
