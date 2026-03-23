"""ID harmonization and heterogeneous graph construction.

Gene ID note: ChG-Miner uses UniProt IDs and DG-AssocMiner uses Entrez IDs.
We prefix them (UNIPROT:xxx and ENTREZ:xxx) and treat them all as "gene" nodes.
They won't overlap, but both pools add structural context. The drug-disease
link via DCh-Miner directly bridges the two halves of the graph.
"""

import os
import pickle
import pandas as pd
import numpy as np
from src.utils import DATA_PROCESSED


def prefix_chg_genes(dtg_df: pd.DataFrame) -> pd.DataFrame:
    """Add UNIPROT: prefix to ChG-Miner gene IDs if not already prefixed."""
    df = dtg_df.copy()
    mask = ~df["gene"].str.startswith("UNIPROT:")
    df.loc[mask, "gene"] = "UNIPROT:" + df.loc[mask, "gene"]
    return df


def build_id_maps(ddi_df: pd.DataFrame, dtg_df: pd.DataFrame,
                  dga_df: pd.DataFrame, dda_df: pd.DataFrame = None):
    """Build global ID mappings for each node type.

    Only drugs that appear in ChCh-Miner (DDI network) are included.
    Genes from both ChG-Miner (UniProt) and DG-AssocMiner (Entrez)
    are pooled under a single "gene" type with distinct prefixed IDs.
    """
    drug_ids = set()
    drug_ids.update(ddi_df["drug1"].unique())
    drug_ids.update(ddi_df["drug2"].unique())

    # Genes: union of UniProt (from ChG-Miner) and Entrez (from DG-AssocMiner)
    gene_ids = set(dtg_df["gene"].unique()) | set(dga_df["gene"].unique())

    # Diseases: union from DG-AssocMiner and optionally DCh-Miner
    disease_ids = set(dga_df["disease"].unique())
    if dda_df is not None:
        disease_ids = disease_ids | set(dda_df["disease"].unique())

    drug_map = {did: i for i, did in enumerate(sorted(drug_ids))}
    gene_map = {gid: i for i, gid in enumerate(sorted(gene_ids))}
    disease_map = {did: i for i, did in enumerate(sorted(disease_ids))}

    id_maps = {"drug": drug_map, "gene": gene_map, "disease": disease_map}

    # Overlap statistics
    dtg_drugs_in_ddi = set(dtg_df["drug"].unique()) & drug_ids
    dga_genes_linked_to_dtg = set(dga_df["gene"].unique()) & set(dtg_df["gene"].unique())

    stats = {
        "num_drugs": len(drug_map),
        "num_genes": len(gene_map),
        "num_diseases": len(disease_map),
        "num_uniprot_genes": sum(1 for g in gene_ids if g.startswith("UNIPROT:")),
        "num_entrez_genes": sum(1 for g in gene_ids if g.startswith("ENTREZ:")),
        "drugs_with_gene_targets": len(dtg_drugs_in_ddi),
        "genes_shared_across_datasets": len(dga_genes_linked_to_dtg),
    }
    if dda_df is not None:
        dda_drugs_in_ddi = set(dda_df["drug"].unique()) & drug_ids
        stats["drugs_with_disease_assoc"] = len(dda_drugs_in_ddi)

    return id_maps, stats


def build_edge_index(df: pd.DataFrame, src_col: str, dst_col: str,
                     src_map: dict, dst_map: dict):
    """Convert a DataFrame of edges to a (2, num_edges) numpy array.

    Filters to only edges where both endpoints are in the ID maps.
    """
    mask = df[src_col].isin(src_map) & df[dst_col].isin(dst_map)
    filtered = df[mask]
    src_idx = filtered[src_col].map(src_map).values.astype(np.int64)
    dst_idx = filtered[dst_col].map(dst_map).values.astype(np.int64)
    edge_index = np.stack([src_idx, dst_idx], axis=0)
    return edge_index


def build_all_edges(ddi_df, dtg_df, dga_df, id_maps, dda_df=None):
    """Build edge indices for all relation types.

    DDI edges are made undirected (both directions stored).
    """
    drug_map = id_maps["drug"]
    gene_map = id_maps["gene"]
    disease_map = id_maps["disease"]

    # Drug-Drug (undirected)
    ddi_edges = build_edge_index(ddi_df, "drug1", "drug2", drug_map, drug_map)
    ddi_rev = np.stack([ddi_edges[1], ddi_edges[0]], axis=0)
    ddi_edges = np.concatenate([ddi_edges, ddi_rev], axis=1)
    ddi_edges = np.unique(ddi_edges, axis=1)

    # Drug-Gene
    dtg_edges = build_edge_index(dtg_df, "drug", "gene", drug_map, gene_map)

    # Disease-Gene
    dga_edges = build_edge_index(dga_df, "disease", "gene", disease_map, gene_map)

    edges = {
        "drug_drug": ddi_edges,
        "drug_gene": dtg_edges,
        "disease_gene": dga_edges,
    }

    # Disease-Drug (optional)
    if dda_df is not None:
        dda_edges = build_edge_index(dda_df, "disease", "drug", disease_map, drug_map)
        edges["disease_drug"] = dda_edges

    for name, ei in edges.items():
        print(f"  [{name}] {ei.shape[1]} edges")

    return edges


def save_processed(id_maps: dict, edges: dict, stats: dict):
    """Save processed graph data to disk."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    with open(os.path.join(DATA_PROCESSED, "id_maps.pkl"), "wb") as f:
        pickle.dump(id_maps, f)
    for name, ei in edges.items():
        np.save(os.path.join(DATA_PROCESSED, f"edges_{name}.npy"), ei)
    with open(os.path.join(DATA_PROCESSED, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print(f"[saved] processed data to {DATA_PROCESSED}")


def load_processed():
    """Load processed graph data from disk."""
    with open(os.path.join(DATA_PROCESSED, "id_maps.pkl"), "rb") as f:
        id_maps = pickle.load(f)
    edges = {}
    for fname in os.listdir(DATA_PROCESSED):
        if fname.startswith("edges_") and fname.endswith(".npy"):
            name = fname[6:-4]
            edges[name] = np.load(os.path.join(DATA_PROCESSED, fname))
    with open(os.path.join(DATA_PROCESSED, "stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    return id_maps, edges, stats
