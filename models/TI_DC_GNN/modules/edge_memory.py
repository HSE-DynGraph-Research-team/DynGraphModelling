import torch

from models.TI_DC_GNN.functions.fusion_fn import get_fusion_fn


class EdgeMemory:
    def __init__(self, n_edges):
        self.n_edges = n_edges
        self.memory = None


class ZeroInitEdgeMemory(EdgeMemory):
    def __init__(self, n_edges, memory_dim):
        super().__init__(n_edges)
        self.memory_dim = memory_dim
        self.memory = torch.zeros((self.n_edges, self.memory_dim))


class NodesInitEdgeMemory(EdgeMemory):
    def __init__(self, n_edges, sources, destinations, node_features, fusion_fn_config):
        super().__init__(n_edges)
        fusion_fn = get_fusion_fn(fn_name=fusion_fn_config['fn_name'],
                                  act_fn_name=fusion_fn_config['fn_name'],
                                  input_dim=node_features.shape[1],
                                  hidden_dim=node_features.shape[1])
        self.memory = fusion_fn(node_features[sources], node_features[destinations])
        self.memory_dim = self.memory.shape[1]


def get_edge_memory(memory_init_name, n_edges, memory_dim, sources, destinations, node_features, fusion_fn_config):
    if memory_init_name == 'nodes':
        return NodesInitEdgeMemory(n_edges, sources, destinations, node_features, fusion_fn_config)
    if memory_init_name == 'zeros':
        return ZeroInitEdgeMemory(n_edges, memory_dim)
