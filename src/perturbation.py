"""Perturbation methods: Gaussian noise and dimensional dropout."""

import torch
import numpy as np


def gaussian_noise_perturbation(embeddings: torch.Tensor, sigma_relative: float,
                                seed: int = None) -> torch.Tensor:
    """Add Gaussian noise scaled relative to the mean embedding norm.

    sigma_absolute = sigma_relative * mean(||z_i||_2)
    """
    if seed is not None:
        torch.manual_seed(seed)
    mean_norm = embeddings.norm(dim=1).mean().item()
    sigma = sigma_relative * mean_norm
    noise = torch.randn_like(embeddings) * sigma
    return embeddings + noise


def dimensional_dropout_perturbation(embeddings: torch.Tensor,
                                     dropout_rate: float,
                                     seed: int = None) -> torch.Tensor:
    """Zero out random dimensions of each embedding.

    Each dimension is independently zeroed with probability dropout_rate.
    Remaining dimensions are NOT rescaled (we want to measure raw sensitivity).
    """
    if seed is not None:
        torch.manual_seed(seed)
    mask = torch.bernoulli(torch.full_like(embeddings, 1.0 - dropout_rate))
    return embeddings * mask


def run_perturbation_trials(embeddings: torch.Tensor, perturb_fn,
                            strengths: list, num_trials: int = 10,
                            base_seed: int = 42):
    """Run multiple perturbation trials for each strength level.

    Args:
        embeddings: original embeddings (num_nodes, embed_dim)
        perturb_fn: gaussian_noise_perturbation or dimensional_dropout_perturbation
        strengths: list of perturbation strength values
        num_trials: number of random trials per strength
        base_seed: base seed (trial seeds = base_seed + trial_idx)

    Returns:
        dict of {strength: [perturbed_embedding_1, ..., perturbed_embedding_n]}
    """
    results = {}
    for strength in strengths:
        trials = []
        for t in range(num_trials):
            seed = base_seed + t
            perturbed = perturb_fn(embeddings, strength, seed=seed)
            trials.append(perturbed)
        results[strength] = trials
    return results
