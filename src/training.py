"""Training loops with early stopping for GCN and RGCN."""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from src.utils import set_seed


def sample_negatives(edge_index, num_nodes, num_neg, seed=None):
    """Sample random negative edges (pairs with no known interaction)."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pos_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    neg_src, neg_dst = [], []
    while len(neg_src) < num_neg:
        s = rng.randint(0, num_nodes)
        d = rng.randint(0, num_nodes)
        if s != d and (s, d) not in pos_set and (d, s) not in pos_set:
            neg_src.append(s)
            neg_dst.append(d)
    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


def _build_labels(num_pos, num_neg, device):
    return torch.cat([torch.ones(num_pos), torch.zeros(num_neg)]).to(device)


def train_gcn(model, optimizer, train_ei, val_ei, graph_ei, num_nodes,
              epochs=200, patience=20, neg_ratio=5, device="cpu", verbose=True):
    """Full training loop for GCN with early stopping.

    Pre-samples fixed validation negatives for stable AUROC measurement.
    """
    best_auroc = 0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    # Fixed validation negatives for stable evaluation
    val_neg = sample_negatives(val_ei, num_nodes, val_ei.shape[1] * neg_ratio, seed=999)
    val_label_index = torch.cat([val_ei, val_neg], dim=1).to(device)
    val_labels = _build_labels(val_ei.shape[1], val_neg.shape[1], device)

    graph_ei_d = graph_ei.to(device)

    pbar = tqdm(range(epochs), disable=not verbose, desc="GCN Training")
    for epoch in pbar:
        # Train with fresh negatives each epoch
        model.train()
        pos_ei = train_ei.to(device)
        num_pos = pos_ei.size(1)
        neg_ei = sample_negatives(train_ei, num_nodes, num_pos * neg_ratio).to(device)
        edge_label_index = torch.cat([pos_ei, neg_ei], dim=1)
        labels = _build_labels(num_pos, neg_ei.size(1), device)

        optimizer.zero_grad()
        scores = model(graph_ei_d, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()
        t_loss = loss.item()

        # Validate with fixed negatives
        model.eval()
        with torch.no_grad():
            v_scores = model(graph_ei_d, val_label_index)
            v_loss = F.binary_cross_entropy_with_logits(v_scores, val_labels).item()
            v_probs = torch.sigmoid(v_scores).cpu().numpy()
            v_auroc = roc_auc_score(val_labels.cpu().numpy(), v_probs)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_auroc"].append(v_auroc)

        pbar.set_postfix(train_loss=f"{t_loss:.4f}", val_auroc=f"{v_auroc:.4f}")

        if v_auroc > best_auroc:
            best_auroc = v_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return history, best_auroc


def train_rgcn(model, optimizer, train_ei, val_ei, graph_ei, graph_et,
               num_drugs, epochs=200, patience=20, neg_ratio=5,
               device="cpu", verbose=True):
    """Full training loop for RGCN with early stopping.

    Pre-samples fixed validation negatives for stable AUROC measurement.
    """
    best_auroc = 0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    # Fixed validation negatives
    val_neg = sample_negatives(val_ei, num_drugs, val_ei.shape[1] * neg_ratio, seed=999)
    val_label_index = torch.cat([val_ei, val_neg], dim=1).to(device)
    val_labels = _build_labels(val_ei.shape[1], val_neg.shape[1], device)

    g_ei = graph_ei.to(device)
    g_et = graph_et.to(device)

    pbar = tqdm(range(epochs), disable=not verbose, desc="RGCN Training")
    for epoch in pbar:
        # Train
        model.train()
        pos_ei = train_ei.to(device)
        num_pos = pos_ei.size(1)
        neg_ei = sample_negatives(train_ei, num_drugs, num_pos * neg_ratio).to(device)
        edge_label_index = torch.cat([pos_ei, neg_ei], dim=1)
        labels = _build_labels(num_pos, neg_ei.size(1), device)

        optimizer.zero_grad()
        scores = model(g_ei, g_et, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()
        t_loss = loss.item()

        # Validate
        model.eval()
        with torch.no_grad():
            v_scores = model(g_ei, g_et, val_label_index)
            v_loss = F.binary_cross_entropy_with_logits(v_scores, val_labels).item()
            v_probs = torch.sigmoid(v_scores).cpu().numpy()
            v_auroc = roc_auc_score(val_labels.cpu().numpy(), v_probs)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_auroc"].append(v_auroc)

        pbar.set_postfix(train_loss=f"{t_loss:.4f}", val_auroc=f"{v_auroc:.4f}")

        if v_auroc > best_auroc:
            best_auroc = v_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return history, best_auroc
