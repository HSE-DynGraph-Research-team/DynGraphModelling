import torch.nn as nn
import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from models.OurTiGCN.model.layers import ConvGCNLayer


class OurTiGCN(nn.Module):

    def __init__(self,
                 nodes_num: int,
                 emb_dim: int,
                 device,
                 neighbour_order_balance_gamma: float,
                 normalize: bool = True,
    ):

        super(OurTiGCN, self).__init__()
        self.nodes_num = nodes_num
        self.emb_dim = emb_dim
        self.device = device
        self.neighbour_order_balance_gamma = neighbour_order_balance_gamma
        self.normalize = normalize
        self.node_embeddings = torch.Tensor(nodes_num, emb_dim).to(self.device).requires_grad_()
        self.freezed_embeddings = None
        self.weight1 = Parameter(torch.Tensor(emb_dim, emb_dim).to(self.device))
        self.weight2 = Parameter(torch.Tensor(emb_dim, emb_dim).to(self.device))
        self.weight_gate = Parameter(torch.Tensor(emb_dim, emb_dim).to(self.device))

        self.gcn_layer = ConvGCNLayer(in_channels=self.emb_dim,
                                      out_channels=self.emb_dim,
                                      normalize=normalize
        )
        self.act = nn.Sigmoid()
        self.init_parameters()
        self.backup_embeddings()
        self.cached_edges = None

    def init_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        glorot(self.weight_gate)

    def backup_embeddings(self):
        self.freezed_embeddings = self.node_embeddings.clone().detach()

    def restore_embeddings(self):
        self.node_embeddings = self.freezed_embeddings.requires_grad_()

    def compute_temporal_embeddings(self, nodes):
        nodes_torch = torch.from_numpy(nodes).to(self.device)
        return self.node_embeddings.index_select(0, nodes_torch)

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes):
        source_nodes_embs = self.compute_temporal_embeddings(source_nodes)
        destination_nodes_embs = self.compute_temporal_embeddings(destination_nodes)
        negative_nodes_embs = self.compute_temporal_embeddings(negative_nodes)
        real_probs = torch.mul(source_nodes_embs, destination_nodes_embs).sum(dim=1).sigmoid()
        negative_probs = torch.mul(source_nodes_embs, negative_nodes_embs).sum(dim=1).sigmoid()
        return real_probs, negative_probs

    def update_cache(self, edges):
        self.cached_edges = edges

    def transform_edges(self, edges):
        unique_edges, counts = edges.unique(return_counts=True, dim=0)
        return unique_edges.T.long(), torch.log(counts + 1)

    def forward(self):
        if self.cached_edges is None:
            return
        edge_index, edge_weight = self.transform_edges(self.cached_edges.to(self.device))
        d_embs1 = self.act(self.gcn_layer.forward(self.node_embeddings, self.weight1, edge_index, edge_weight))
        d_embs2 = self.act(self.gcn_layer.forward(d_embs1, self.weight1, edge_index, edge_weight))
        d_embs_combine = self.neighbour_order_balance_gamma * d_embs1 + \
                         (1 - self.neighbour_order_balance_gamma) * d_embs2
        d_embs_combine_gated = self.act(torch.matmul(d_embs_combine, self.weight_gate))
        d_embs = torch.mul(d_embs_combine_gated, d_embs_combine)
        self.node_embeddings = self.node_embeddings + d_embs
