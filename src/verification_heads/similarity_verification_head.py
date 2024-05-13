import torch

from src.custom_distances import VerificationSimilarityBase, LpSimilarity, VerificationCosineSimilarity
from src.utils.embeddings import Embeddings
from src.verification_heads.verification_head_base import VerificationHeadBase


class SimilarityVerificationHead(VerificationHeadBase):
    def __init__(self, distance: VerificationSimilarityBase = None, threshold=None):
        distance = LpSimilarity() if distance is None else distance
        super().__init__(distance=distance, threshold=threshold)

    def _is_normalized(self) -> bool:
        return True

    def forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, *_args):
        return self.distance.transform_to_similarity_between_0_and_1(
            self._distance_forward_in_batches(query_embeddings, reference_embeddings)
        )

    def get_matches(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor,
                    threshold=None, n_closest=None) -> torch.tensor:
        predictions = self.forward(query_embeddings, reference_embeddings).ge(
            (threshold or self.best_threshold)).float()
        return self.apply_n_closest(predictions, n_closest)


if __name__ == '__main__':
    queries = torch.randn((2, 2))
    references = torch.randn(3, 2)
    head = SimilarityVerificationHead(VerificationCosineSimilarity())
    print(head(queries, references))
    head = SimilarityVerificationHead(LpSimilarity())
    print(head(queries, references))
    head = SimilarityVerificationHead()
    print(head(queries, references))

    embeddings = Embeddings(
        query=torch.randn((100, 10)),
        query_labels=torch.randint(0, 9, (100,)),
        reference=torch.randn((200, 10)),
        reference_labels=torch.randint(0, 9, (200,))
    )
    preds, gt, reshaped_refs = head.compare_groupwise(embeddings)
    for user in reshaped_refs.unique():
        user_mask = reshaped_refs == user
        print(preds[:, user_mask].size())
        print(gt[:, user_mask].size())
