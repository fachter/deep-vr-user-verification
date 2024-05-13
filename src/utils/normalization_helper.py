from torch.nn.functional import normalize


def normalize_embedding(x, dim=0):
    normalized_data = normalize(x, dim=dim)
    return normalized_data
