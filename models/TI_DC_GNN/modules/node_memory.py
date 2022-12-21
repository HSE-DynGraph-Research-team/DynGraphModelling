import torch


class NodeMemory:
    def __init__(self, n_nodes, memory_dim):
        self.n_nodes = n_nodes
        self.memory_dim = memory_dim
        self.memory = torch.zeros((self.n_nodes, self.memory_dim))


def get_node_memory(n_nodes, memory_dim):
    return NodeMemory(n_nodes, memory_dim)
