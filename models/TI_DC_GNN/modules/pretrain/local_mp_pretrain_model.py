import torch
import torch.nn as nn
import numpy as np

from .decoder import DualNodeSimilarityDecoder, get_node_edge_decoder


class LocalMPModelPretrain(nn.Module):
    def __init__(self, node_features, edge_features, device,
                 edge_memory, node_memory, local_mp, time_encoding, decoder_node_sim_config, decoder_node_edge_sim_config):
        super().__init__()

        # device
        self.device = device
        # features, dimensions, statistics
        self.node_features = node_features
        self.edge_features = edge_features
        self.n_nodes = self.node_features.shape[0]
        self.node_features_dim = self.node_features.shape[1]
        self.n_edges = self.edge_features.shape[0]
        self.edge_features_dim = self.edge_features.shape[1]

        # edge memory
        self.edge_memory = edge_memory
        self.edge_memory_dim = edge_memory.memory_dim
        self.edge_memory_dim = self.edge_memory.memory_dim

        # node memory
        self.node_memory = node_memory
        self.node_memory_dim = node_memory.memory_dim
        # local mp
        self.local_mp = local_mp
        # time encoding
        self.time_encoding = time_encoding
        # decoders for scoring
        if decoder_node_sim_config['use_decoder']:
            self.decoder_node_similarity = DualNodeSimilarityDecoder(self.edge_memory_dim)
        else:
            self.decoder_node_similarity = None
        if decoder_node_edge_sim_config['use_decoder']:
            self.decoder_node_edge_similarity = get_node_edge_decoder(decoder_name=decoder_node_edge_sim_config['name'],
                                                                      node_memory_dim=self.edge_memory_dim,
                                                                      edge_memory_dim=self.node_memory_dim,
                                                                      add_bias=decoder_node_edge_sim_config['add_bias'])
        else:
            self.decoder_node_edge_similarity = None



    def pretrain(self):




