import torch

from models.TI_GNN.layers.fusion_fn import get_fusion_fn


class EdgeMemory:
    def __init__(self, n_edges, memory_dim):
        self.n_edges = n_edges
        self.memory_dim = memory_dim
        self.memory = None


class ZeroInitEdgeMemory(EdgeMemory):
    def __init__(self, n_edges, memory_dim):
        super().__init__(n_edges, memory_dim)
        self.memory = torch.zeros((self.n_edges, self.memory_dim))


class NodesInitEdgeMemory(EdgeMemory):
    def __init__(self, n_edges, memory_dim, sources, destinations, node_features, fusion_fn_name):
        super().__init__(n_edges, memory_dim)
        fusion_fn = get_fusion_fn(fusion_fn_name)
        self.memory = fusion_fn(node_features[sources], node_features[destinations])
        self.memory_dim = self.memory.shape[1]


def get_edge_memory(memory_init_name, n_edges, memory_dim, sources, destinations, node_features, fusion_fn_name):
    if memory_init_name == 'nodes':
        return NodesInitEdgeMemory(n_edges, memory_dim, sources, destinations, node_features, fusion_fn_name)
    if memory_init_name == 'zeros':
        return ZeroInitEdgeMemory(n_edges, memory_dim)
