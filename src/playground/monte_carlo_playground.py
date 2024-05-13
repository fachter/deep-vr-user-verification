import torch
from time import time


# %%

# Step 1: Define your model
def sample_from_model(value: torch.tensor):
    # Replace this with your own model or sampling logic
    cut = value.size(1) // 2
    mean = value[:, :cut]
    std_dev = value[:, cut:]
    return torch.normal(mean, std_dev)


for _ in range(8):
    print(sample_from_model(torch.tensor([
        [1., 0.5, 0.5, 0.5],
        [2., 1.5, 0.1, 0.1],
    ])))

# %%

import torch
import torch.distributions as dist

# Mean and variance vectors
mean = torch.tensor([0.0, 1.0, -1.0])  # Replace with your actual mean vector
variance = torch.tensor([1.0, 0.5, 2.0])  # Replace with your actual variance vector

# Create a Gaussian distribution
gaussian_dist = dist.Normal(mean, torch.sqrt(variance))

# Number of samples
num_samples = 10

# Sample from the distribution
samples = gaussian_dist.sample((num_samples,))

print("Generated samples:")
print(samples)
