"""Build PyTorch Geometric HeteroData objects from processed data."""

import torch
import numpy as np
from torch_geometric.data import HeteroData


def build_hetero_data(id_maps: dict, edges: dict) -> HeteroData:
    """Construct a HeteroData object from ID maps and edge arrays.

    Node features are learnable embeddings (handled in the model),
    so we only store node counts and edge indices here.
    """
    data = HeteroData()

    data["drug"].num_nodes = len(id_maps["drug"])
    data["gene"].num_nodes = len(id_maps["gene"])
    data["disease"].num_nodes = len(id_maps["disease"])

    if "drug_drug" in edges:
        ei = torch.from_numpy(edges["drug_drug"]).long()
        data["drug", "interacts", "drug"].edge_index = ei

    if "drug_gene" in edges:
        ei = torch.from_numpy(edges["drug_gene"]).long()
        data["drug", "targets", "gene"].edge_index = ei
        rev = torch.stack([ei[1], ei[0]], dim=0)
        data["gene", "targeted_by", "drug"].edge_index = rev

    if "disease_gene" in edges:
        ei = torch.from_numpy(edges["disease_gene"]).long()
        data["disease", "associated_with", "gene"].edge_index = ei
        rev = torch.stack([ei[1], ei[0]], dim=0)
        data["gene", "associated_with", "disease"].edge_index = rev

    if "disease_drug" in edges:
        ei = torch.from_numpy(edges["disease_drug"]).long()
        data["disease", "treated_by", "drug"].edge_index = ei
        rev = torch.stack([ei[1], ei[0]], dim=0)
        data["drug", "treats", "disease"].edge_index = rev

    return data


def build_homo_data(id_maps: dict, edges: dict):
    """Build a simple homogeneous graph with only drug-drug edges.

    Returns (edge_index, num_nodes) for the GCN baseline.
    """
    num_drugs = len(id_maps["drug"])
    ei = torch.from_numpy(edges["drug_drug"]).long()
    return ei, num_drugs
