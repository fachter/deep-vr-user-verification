from dataclasses import dataclass
import torch


@dataclass
class Embeddings:
    query: torch.tensor
    reference: torch.tensor
    query_labels: torch.tensor
    reference_labels: torch.tensor
