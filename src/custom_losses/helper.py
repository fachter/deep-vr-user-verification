import torch


def loss_soft_con(probabilities: torch.tensor, ground_truth: torch.tensor):
    log_input = (1. - ground_truth) * (1. - probabilities) + ground_truth * probabilities
    clipped_inputs = torch.clip(log_input, 1e-10)
    loss = -torch.log(clipped_inputs)
    return loss
