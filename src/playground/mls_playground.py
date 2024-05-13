import torch
import numpy as np


# %%

def logmls(parameters1, parameters2):
    """Compute Log Mutual Likelihood Score (MLS) for pairs of distributions.


    Args:
        parameters1: Distribution parameters with shape (..., K).
        parameters2: Distribution parameters with shape (..., K).

    Returns:
        MLS scores with shape (...).
    """
    log_probs1, means1, hidden_vars1 = self.split_parameters(parameters1)  # (..., C), (..., C, D), (..., C, D).
    log_probs2, means2, hidden_vars2 = self.split_parameters(parameters2)  # (..., C), (..., C, D), (..., C, D).
    logvars1 = self._parametrization.log_positive(hidden_vars1)
    logvars2 = self._parametrization.log_positive(hidden_vars2)
    pairwise_logmls = _normal_logmls(
        means1=means1[..., :, None, :],  # (..., C, 1, D).
        logvars1=logvars1[..., :, None, :],  # (..., C, 1, D).
        means2=means2[..., None, :, :],  # (..., 1, C, D).
        logvars2=logvars2[..., None, :, :]  # (..., 1, C, D).
    )  # (..., C, C).
    pairwise_logprobs = log_probs1[..., :, None] + log_probs2[..., None, :]  # (..., C, C).
    dim_prefix = list(pairwise_logmls.shape)[:-2]
    logmls = torch.logsumexp((pairwise_logprobs + pairwise_logmls).reshape(*(dim_prefix + [-1])), dim=-1)  # (...).
    return logmls


_config = {
    'dim': 2
}


def _normal_logmls(means1, logvars1, means2, logvars2):
    """Compute Log MLS for unimodal distributions.

    For implementation details see "Probabilistic Face Embeddings":
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.pdf
    """
    c = -0.5 * _config["dim"] * np.log(2 * np.pi)
    delta2 = torch.square(means1 - means2)  # (..., D).
    covsum = logvars1.exp() + logvars2.exp()  # (..., D).
    logcovsum = torch.logaddexp(logvars1, logvars2)  # (..., D).
    mls = c - 0.5 * (delta2 / covsum + logcovsum).sum(-1)  # (...).
    return mls


res = _normal_logmls(torch.tensor([1., 0.5]), torch.tensor([0.5, 0.5]), torch.tensor([1.2, 1.1]),
                     torch.tensor([0.8, 1.0]))
