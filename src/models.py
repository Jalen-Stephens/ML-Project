"""GCN baseline, RGCN model, and heuristic baseline for DDI link prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, RGCNConv


class LinkDecoder(nn.Module):
    """Dot-product decoder for link prediction."""

    def forward(self, z_src, z_dst):
        return (z_src * z_dst).sum(dim=-1)

    def forward_all(self, z):
        """Score all pairs: returns (num_nodes, num_nodes) score matrix."""
        return z @ z.t()


class GCNLinkPredictor(nn.Module):
    """2-layer GCN on the homogeneous drug-drug graph."""

    def __init__(self, num_nodes: int, embed_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        self.conv1 = GCNConv(embed_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, embed_dim)
        self.decoder = LinkDecoder()
        self.dropout = dropout
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def encode(self, edge_index):
        x = self.embedding.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return self.decoder(z[src], z[dst])

    def forward(self, edge_index, edge_label_index):
        z = self.encode(edge_index)
        return self.decode(z, edge_label_index)


class RGCNLinkPredictor(nn.Module):
    """2-layer RGCN on the full heterogeneous graph.

    Operates on a flattened (homogeneous) view where node types are
    concatenated and edge types are encoded via relation indices.
    """

    def __init__(self, num_nodes_dict: dict, embed_dim: int = 64,
                 num_relations: int = 4, num_bases: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.node_types = sorted(num_nodes_dict.keys())
        self.num_nodes_dict = num_nodes_dict
        total_nodes = sum(num_nodes_dict.values())

        self.embeddings = nn.ModuleDict({
            ntype: nn.Embedding(num_nodes_dict[ntype], embed_dim)
            for ntype in self.node_types
        })
        self.conv1 = RGCNConv(embed_dim, embed_dim,
                              num_relations=num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(embed_dim, embed_dim,
                              num_relations=num_relations, num_bases=num_bases)
        self.decoder = LinkDecoder()
        self.dropout = dropout

        self._node_offsets = {}
        offset = 0
        for ntype in self.node_types:
            self._node_offsets[ntype] = offset
            offset += num_nodes_dict[ntype]

        self._init_weights()

    def _init_weights(self):
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

    @property
    def drug_offset(self):
        return self._node_offsets["drug"]

    @property
    def num_drugs(self):
        return self.num_nodes_dict["drug"]

    def get_initial_embeddings(self):
        """Concatenate all node-type embeddings into a single tensor."""
        parts = [self.embeddings[ntype].weight for ntype in self.node_types]
        return torch.cat(parts, dim=0)

    def encode(self, edge_index, edge_type):
        x = self.get_initial_embeddings()
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

    def get_drug_embeddings(self, z):
        """Extract drug embeddings from the full node embedding tensor."""
        start = self.drug_offset
        end = start + self.num_drugs
        return z[start:end]

    def decode(self, z, edge_label_index):
        z_drug = self.get_drug_embeddings(z)
        src, dst = edge_label_index
        return self.decoder(z_drug[src], z_drug[dst])

    def forward(self, edge_index, edge_type, edge_label_index):
        z = self.encode(edge_index, edge_type)
        return self.decode(z, edge_label_index)


def flatten_hetero_graph(hetero_data, node_types_order, relation_map):
    """Convert HeteroData edges to flat edge_index + edge_type tensors.

    Args:
        hetero_data: PyG HeteroData
        node_types_order: sorted list of node types (must match RGCNLinkPredictor)
        relation_map: dict mapping (src_type, rel, dst_type) -> int relation id

    Returns:
        edge_index (2, total_edges), edge_type (total_edges,), node_offsets dict
    """
    offsets = {}
    offset = 0
    for ntype in node_types_order:
        offsets[ntype] = offset
        offset += hetero_data[ntype].num_nodes

    all_src, all_dst, all_types = [], [], []
    for (s_type, rel, d_type), rel_id in relation_map.items():
        key = (s_type, rel, d_type)
        if key in hetero_data.edge_types:
            ei = hetero_data[key].edge_index
            all_src.append(ei[0] + offsets[s_type])
            all_dst.append(ei[1] + offsets[d_type])
            all_types.append(torch.full((ei.size(1),), rel_id, dtype=torch.long))

    edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
    edge_type = torch.cat(all_types)
    return edge_index, edge_type, offsets


# -- Relation type mapping (consistent ordering) --
RELATION_MAP = {
    ("drug", "interacts", "drug"): 0,
    ("drug", "targets", "gene"): 1,
    ("gene", "targeted_by", "drug"): 2,
    ("disease", "associated_with", "gene"): 3,
    ("gene", "associated_with", "disease"): 4,
    ("disease", "treated_by", "drug"): 5,
    ("drug", "treats", "disease"): 6,
}

NODE_TYPES_ORDER = ["disease", "drug", "gene"]


def compute_common_neighbor_scores(edge_index, num_nodes):
    """Heuristic baseline: common neighbor count for all drug pairs.

    Returns a sparse-ish score via adjacency multiplication.
    Only practical for small graphs (< 5000 nodes).
    """
    from scipy.sparse import coo_matrix

    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    adj = coo_matrix(
        (np.ones(len(src)), (src, dst)),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    cn_matrix = (adj @ adj).toarray()
    np.fill_diagonal(cn_matrix, 0)
    return cn_matrix
