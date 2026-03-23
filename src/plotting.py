"""All matplotlib figure functions for exploratory and final plots."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
})

COLORS = {"gcn": "#1f77b4", "rgcn": "#ff7f0e", "heuristic": "#2ca02c"}


def _save(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


# --- E1: Degree distribution ---
def plot_degree_distribution(degrees, title="Drug Degree Distribution", save_path=None):
    fig, ax = plt.subplots()
    ax.hist(degrees, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_yscale("log")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- E2: Venn-style overlap bar chart ---
def plot_id_overlap(overlap_counts: dict, title="Drug ID Overlap", save_path=None):
    fig, ax = plt.subplots()
    names = list(overlap_counts.keys())
    counts = list(overlap_counts.values())
    bars = ax.barh(names, counts, color=sns.color_palette("Set2", len(names)))
    ax.set_xlabel("Number of Drugs")
    ax.set_title(title)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- E4/E5: Training curves ---
def plot_training_curves(histories: dict, metric="train_loss", save_path=None):
    fig, ax = plt.subplots()
    for name, hist in histories.items():
        ax.plot(hist[metric], label=name, color=COLORS.get(name))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Training: {metric.replace('_', ' ').title()}")
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- E6: ROC curves ---
def plot_roc_curves(curves_dict: dict, aurocs: dict = None, save_path=None):
    fig, ax = plt.subplots()
    for name, c in curves_dict.items():
        label = name
        if aurocs and name in aurocs:
            label = f"{name} (AUROC={aurocs[name]:.3f})"
        ax.plot(c["fpr"], c["tpr"], label=label, color=COLORS.get(name))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- E7: PR curves ---
def plot_pr_curves(curves_dict: dict, auprcs: dict = None, save_path=None):
    fig, ax = plt.subplots()
    for name, c in curves_dict.items():
        label = name
        if auprcs and name in auprcs:
            label = f"{name} (AUPRC={auprcs[name]:.3f})"
        ax.plot(c["recall"], c["precision"], label=label, color=COLORS.get(name))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- E8/Figure 2: Stability Score vs perturbation strength ---
def plot_stability_curves(results: dict, save_path=None):
    """
    results: {model_name: {strength: {"stability_score_mean": ..., "stability_score_std": ...}}}
    """
    fig, ax = plt.subplots()
    for name, data in results.items():
        strengths = sorted(data.keys())
        means = [data[s]["stability_score_mean"] for s in strengths]
        stds = [data[s]["stability_score_std"] for s in strengths]
        means, stds = np.array(means), np.array(stds)
        ax.plot(strengths, means, "o-", label=name, color=COLORS.get(name))
        ax.fill_between(strengths, means - stds, means + stds,
                        alpha=0.15, color=COLORS.get(name))
    ax.set_xlabel("Perturbation Strength")
    ax.set_ylabel("Stability Score")
    ax.set_title("Stability Under Perturbation")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- E10/Figure 3: Top-K Jaccard vs perturbation strength ---
def plot_topk_overlap(results: dict, ks=(10, 20, 50), save_path=None):
    """
    results: {model_name: {strength: {f"mean_jaccard_top{k}_mean": ...}}}
    """
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 5), sharey=True)
    if len(ks) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks):
        for name, data in results.items():
            strengths = sorted(data.keys())
            means = [data[s][f"mean_jaccard_top{k}_mean"] for s in strengths]
            ax.plot(strengths, means, "o-", label=name, color=COLORS.get(name))
        ax.set_xlabel("Perturbation Strength")
        ax.set_ylabel(f"Mean Jaccard (Top-{k})")
        ax.set_title(f"Top-{k} Overlap")
        ax.set_ylim(0, 1.05)
        ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- E9: Mean |delta p| vs perturbation strength ---
def plot_delta_p_curves(results: dict, save_path=None):
    fig, ax = plt.subplots()
    for name, data in results.items():
        strengths = sorted(data.keys())
        means = [data[s]["mean_delta_p_mean"] for s in strengths]
        stds = [data[s]["mean_delta_p_std"] for s in strengths]
        means, stds = np.array(means), np.array(stds)
        ax.plot(strengths, means, "o-", label=name, color=COLORS.get(name))
        ax.fill_between(strengths, means - stds, means + stds,
                        alpha=0.15, color=COLORS.get(name))
    ax.set_xlabel("Perturbation Strength")
    ax.set_ylabel("Mean |Δp|")
    ax.set_title("Prediction Probability Shift")
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- E11: Spearman rho vs perturbation strength ---
def plot_spearman_curves(results: dict, save_path=None):
    fig, ax = plt.subplots()
    for name, data in results.items():
        strengths = sorted(data.keys())
        means = [data[s]["mean_spearman_rho_mean"] for s in strengths]
        ax.plot(strengths, means, "o-", label=name, color=COLORS.get(name))
    ax.set_xlabel("Perturbation Strength")
    ax.set_ylabel("Mean Spearman ρ")
    ax.set_title("Ranking Correlation Under Perturbation")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- E12: Box plot of per-drug delta p ---
def plot_delta_p_boxplot(per_drug_deltas: dict, save_path=None):
    fig, ax = plt.subplots()
    data_list = [per_drug_deltas[name] for name in per_drug_deltas]
    labels = list(per_drug_deltas.keys())
    bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
    for patch, name in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS.get(name, "#999999"))
        patch.set_alpha(0.6)
    ax.set_ylabel("Mean |Δp| per Drug")
    ax.set_title("Per-Drug Prediction Shift Distribution (σ=0.10)")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- E16/E17/Figure 4: Edge-type ablation grouped bar chart ---
def plot_ablation_bars(ablation_results: dict, metric_keys: list,
                       title="Edge-Type Ablation", save_path=None):
    """
    ablation_results: {variant_name: {metric: value}}
    metric_keys: list of metric names to plot as grouped bars
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    variants = list(ablation_results.keys())
    x = np.arange(len(variants))
    width = 0.8 / len(metric_keys)
    for i, mk in enumerate(metric_keys):
        vals = [ablation_results[v].get(mk, 0) for v in variants]
        ax.bar(x + i * width, vals, width, label=mk)
    ax.set_xticks(x + width * (len(metric_keys) - 1) / 2)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- E14/E15/Figure 5: Cold-start comparison ---
def plot_coldstart_comparison(standard_metrics: dict, coldstart_metrics: dict,
                              metric_keys: list, save_path=None):
    fig, ax = plt.subplots()
    x = np.arange(len(metric_keys))
    width = 0.35
    std_vals = [standard_metrics.get(m, 0) for m in metric_keys]
    cold_vals = [coldstart_metrics.get(m, 0) for m in metric_keys]
    ax.bar(x - width / 2, std_vals, width, label="Standard", color=COLORS["gcn"])
    ax.bar(x + width / 2, cold_vals, width, label="Cold-Start", color=COLORS["rgcn"])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=15, ha="right")
    ax.set_title("Standard vs Cold-Start Evaluation")
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# --- Figure 6: Summary results table ---
def plot_results_table(df, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("Summary Results", fontsize=14, pad=20)
    plt.tight_layout()
    _save(fig, save_path)
    return fig
