"""Evaluation metrics: AUROC, AUPRC, Hits@K, MRR."""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


@torch.no_grad()
def compute_link_scores(model, edge_index, model_type="gcn",
                        graph_edge_index=None, graph_edge_type=None,
                        device="cpu"):
    """Get raw logit scores for a set of edges."""
    model.eval()
    if model_type == "gcn":
        z = model.encode(graph_edge_index.to(device))
        scores = model.decode(z, edge_index.to(device))
    else:
        z = model.encode(graph_edge_index.to(device),
                         graph_edge_type.to(device))
        scores = model.decode(z, edge_index.to(device))
    return scores.cpu()


def compute_metrics(pos_scores, neg_scores):
    """Compute AUROC, AUPRC from positive and negative scores."""
    scores = torch.cat([pos_scores, neg_scores]).numpy()
    labels = np.concatenate([np.ones(len(pos_scores)),
                             np.zeros(len(neg_scores))])
    probs = 1 / (1 + np.exp(-scores))  # sigmoid
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    return {"auroc": auroc, "auprc": auprc}


def compute_curves(pos_scores, neg_scores):
    """Compute ROC and PR curve data for plotting."""
    scores = torch.cat([pos_scores, neg_scores]).numpy()
    labels = np.concatenate([np.ones(len(pos_scores)),
                             np.zeros(len(neg_scores))])
    probs = 1 / (1 + np.exp(-scores))
    fpr, tpr, _ = roc_curve(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}


@torch.no_grad()
def compute_ranking_metrics(model, test_pos_ei, num_drugs, model_type="gcn",
                            graph_edge_index=None, graph_edge_type=None,
                            device="cpu", ks=(10, 20, 50)):
    """Compute Hits@K and MRR for test positive edges.

    For each positive edge (u, v), rank v among all drugs for u.
    """
    model.eval()
    if model_type == "gcn":
        z = model.encode(graph_edge_index.to(device))
    else:
        z_full = model.encode(graph_edge_index.to(device),
                              graph_edge_type.to(device))
        z = model.get_drug_embeddings(z_full)

    z = z.cpu()
    all_scores = z @ z.t()  # (num_drugs, num_drugs)

    ranks = []
    pos_src = test_pos_ei[0].numpy()
    pos_dst = test_pos_ei[1].numpy()
    for i in range(len(pos_src)):
        u, v = pos_src[i], pos_dst[i]
        scores_u = all_scores[u].clone()
        scores_u[u] = -1e9  # exclude self
        rank = (scores_u > scores_u[v]).sum().item() + 1
        ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float64)
    metrics = {"mrr": float(np.mean(1.0 / ranks))}
    for k in ks:
        metrics[f"hits@{k}"] = float(np.mean(ranks <= k))
    return metrics
