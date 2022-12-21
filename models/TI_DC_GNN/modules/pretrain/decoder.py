from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class DualNodeSimilarityDecoder(nn.Module):
    def __init__(self, memory_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(memory_dim, memory_dim))
        self.act = nn.Sigmoid()
        nn.init.xavier_uniform_(self.weight)

    def forward(self, sources, destinations):
        intermediate = torch.matmul(sources, self.weight)
        return self.act(torch.matmul(intermediate, torch.transpose(destinations, 0, 1)))


class DualNodeEdgeDecoder(nn.Module, ABC):
    def __init__(self,
                 node_memory_dim,
                 edge_memory_dim,
                 add_bias=False):
        super().__init__()
        self.node_memory_dim = node_memory_dim
        self.edge_memory_dim = edge_memory_dim
        self.add_bias = add_bias
        self.build_model()

    @abstractmethod
    def build_model(self):
        pass

    def calculate_scores(self, source_memory_batch, edge_memory_batch, dest_memory_batch):
        return torch.sum(source_memory_batch * edge_memory_batch * dest_memory_batch, dim=1)

    def prepare_node_memory(self, node_memory):
        return node_memory

    def prepare_edge_memory(self, edge_memory):
        return edge_memory

    def forward(self, source_memory_batch, edge_memory_batch, dest_memory_batch):
        source_memory_batch = self.prepare_node_memory(source_memory_batch)
        edge_memory_batch = self.prepare_edge_memory(edge_memory_batch)
        dest_memory_batch = self.prepare_node_memory(dest_memory_batch)
        return self.calculate_scores(source_memory_batch, edge_memory_batch, dest_memory_batch)


class EdgeTransformDecoder(DualNodeEdgeDecoder):
    def build_model(self):
        self.linear_transform = nn.Linear(self.edge_memory_dim, self.node_memory_dim, bias=self.add_bias)

    def prepare_edge_memory(self, edge_memory_dim):
        return self.linear_transform(edge_memory_dim)


class NodeTransformDecoder(DualNodeEdgeDecoder):
    def build_model(self):
        self.linear_transform = nn.Linear(self.node_memory_dim, self.edge_memory_dim, bias=self.add_bias)

    def prepare_node_memory(self, node_memory):
        return self.linear_transform(node_memory)


def get_node_edge_decoder(decoder_name, node_memory_dim, edge_memory_dim, add_bias):
    if decoder_name == 'node_transform':
        return NodeTransformDecoder(node_memory_dim, edge_memory_dim, add_bias)
    elif decoder_name == 'edge_transform':
        return EdgeTransformDecoder(node_memory_dim, edge_memory_dim, add_bias)
