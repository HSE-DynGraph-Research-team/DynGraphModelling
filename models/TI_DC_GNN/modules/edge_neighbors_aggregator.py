import torch
import torch.nn as nn


class EdgeNeighborsAggregator(nn.Module):
    def __init__(self, edge_features_dim, n_neighbors, device):
        self.edge_features_dim = edge_features_dim
        self.n_neighbors = n_neighbors
        self.device = device

    def get_cur_features(self, edge_features, neighbors_edge_idxs):
        # neighbors_edge_idxs: [nxk]
        # edge_features: [Nxd]
        # returns [nxkxd]
        return edge_features[neighbors_edge_idxs].to(self.device)

    def forward(self, edge_features, neighbors_edge_idxs):
        pass


class MeanEdgeNeighborsAggregator(EdgeNeighborsAggregator):
    def forward(self, edge_features, neighbors_edge_idxs):
        cur_edge_features = self.get_cur_features(edge_features, neighbors_edge_idxs)   # [nxkxd]
        return torch.mean(cur_edge_features, dim=1)


class MLPEdgeNeighborsAggregator(EdgeNeighborsAggregator):
    def __init__(self, edge_features_dim, n_neighbors, device):
        super().__init__(edge_features_dim, n_neighbors, device)
        self.edge_features_dim = edge_features_dim
        self.n_neighbors = n_neighbors
        self.device = device
        self.fc = nn.Linear(edge_features_dim * n_neighbors, edge_features_dim)

    def forward(self, edge_features, neighbors_edge_idxs):
        cur_edge_features = self.get_cur_features(edge_features, neighbors_edge_idxs)   # [nxkxd]
        cur_edge_features.view(cur_edge_features.shape[0], self.n_neighbors*self.edge_features_dim)
        return self.fc(cur_edge_features)


def get_edge_neighbors_aggregator(agg_name, edge_features_dim, n_neighbors, device):
    if agg_name == 'mean':
        return MeanEdgeNeighborsAggregator(edge_features_dim, n_neighbors, device)
    if agg_name == 'mlp':
        return MLPEdgeNeighborsAggregator(edge_features_dim, n_neighbors, device)
