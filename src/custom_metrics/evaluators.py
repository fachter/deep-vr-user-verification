from dataclasses import dataclass
from typing import Any

from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from src.custom_metrics.verification_evaluator import VerificationEvaluator


@dataclass
class Evaluators:
    identification: AccuracyCalculator
    verification: VerificationEvaluator
