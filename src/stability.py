"""Stability metrics: prediction shift, ranking shift, top-K Jaccard, Stability Score."""

import torch
import numpy as np
from scipy.stats import spearmanr


def prediction_probability_shift(z_original, z_perturbed, eval_pairs=None):
    """Compute |delta p| for each evaluated drug pair.

    If eval_pairs is None, computes for all drug pairs (expensive for large graphs).
    Otherwise eval_pairs is a (2, N) tensor of (src, dst) indices.

    Returns dict with mean, median, max, std of |delta p|.
    """
    with torch.no_grad():
        if eval_pairs is not None:
            src, dst = eval_pairs
            orig_scores = torch.sigmoid((z_original[src] * z_original[dst]).sum(dim=-1))
            pert_scores = torch.sigmoid((z_perturbed[src] * z_perturbed[dst]).sum(dim=-1))
        else:
            orig_scores = torch.sigmoid(z_original @ z_original.t())
            pert_scores = torch.sigmoid(z_perturbed @ z_perturbed.t())

        delta_p = (orig_scores - pert_scores).abs()

        if eval_pairs is None:
            n = z_original.size(0)
            mask = ~torch.eye(n, dtype=torch.bool)
            delta_p = delta_p[mask]

        return {
            "mean_delta_p": delta_p.mean().item(),
            "median_delta_p": delta_p.median().item(),
            "max_delta_p": delta_p.max().item(),
            "std_delta_p": delta_p.std().item(),
        }


def ranking_shift(z_original, z_perturbed):
    """Compute mean Spearman rho of per-drug rankings before/after perturbation."""
    with torch.no_grad():
        orig_scores = (z_original @ z_original.t()).cpu().numpy()
        pert_scores = (z_perturbed @ z_perturbed.t()).cpu().numpy()

    n = orig_scores.shape[0]
    np.fill_diagonal(orig_scores, -np.inf)
    np.fill_diagonal(pert_scores, -np.inf)

    rhos = []
    for i in range(n):
        rho, _ = spearmanr(orig_scores[i], pert_scores[i])
        if not np.isnan(rho):
            rhos.append(rho)

    return {
        "mean_spearman_rho": float(np.mean(rhos)),
        "std_spearman_rho": float(np.std(rhos)),
    }


def topk_jaccard(z_original, z_perturbed, k=20):
    """Compute mean Jaccard similarity of top-K predicted partners per drug."""
    with torch.no_grad():
        orig_scores = z_original @ z_original.t()
        pert_scores = z_perturbed @ z_perturbed.t()
        orig_scores.fill_diagonal_(-1e9)
        pert_scores.fill_diagonal_(-1e9)

        orig_topk = orig_scores.topk(k, dim=1).indices.cpu().numpy()
        pert_topk = pert_scores.topk(k, dim=1).indices.cpu().numpy()

    jaccards = []
    for i in range(orig_topk.shape[0]):
        orig_set = set(orig_topk[i])
        pert_set = set(pert_topk[i])
        j = len(orig_set & pert_set) / len(orig_set | pert_set)
        jaccards.append(j)

    return {
        f"mean_jaccard_top{k}": float(np.mean(jaccards)),
        f"std_jaccard_top{k}": float(np.std(jaccards)),
    }


def compute_stability_score(mean_delta_p, mean_spearman_rho, mean_jaccard_top20):
    """Composite Stability Score in [0, 1] where 1 = perfectly stable.

    StabilityScore = (1/3) * [(1 - mean_delta_p) + mean_spearman_rho + mean_jaccard_top20]
    """
    pred_stability = max(0.0, 1.0 - mean_delta_p)
    rank_stability = max(0.0, mean_spearman_rho)
    topk_stability = max(0.0, mean_jaccard_top20)
    return (pred_stability + rank_stability + topk_stability) / 3.0


def full_stability_eval(z_original, z_perturbed, eval_pairs=None, ks=(10, 20, 50)):
    """Run all stability metrics for one perturbation trial.

    Returns a flat dict of all metrics including the composite Stability Score.
    """
    metrics = {}
    ps = prediction_probability_shift(z_original, z_perturbed, eval_pairs)
    metrics.update(ps)

    rs = ranking_shift(z_original, z_perturbed)
    metrics.update(rs)

    for k in ks:
        jk = topk_jaccard(z_original, z_perturbed, k=k)
        metrics.update(jk)

    metrics["stability_score"] = compute_stability_score(
        metrics["mean_delta_p"],
        metrics["mean_spearman_rho"],
        metrics["mean_jaccard_top20"],
    )
    return metrics


def aggregate_trial_metrics(trial_metrics_list):
    """Aggregate metrics across multiple perturbation trials.

    Returns dict with mean and std for each metric.
    """
    keys = trial_metrics_list[0].keys()
    agg = {}
    for key in keys:
        vals = [m[key] for m in trial_metrics_list]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    return agg
