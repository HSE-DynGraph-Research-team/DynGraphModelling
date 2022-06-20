import torch
import torch.nn as nn
import numpy as np
from .time_encoding import TimeEncode
from models.TI_GNN.layers.mlp import MergeNodesLayer
from ..modules.edge_memory import get_edge_memory
from ..modules.edge_neighbors_aggregator import get_edge_neighbors_aggregator
from ..modules.mail_box import MailBox
from ..modules.ti_message_passing.ti_local_message_passing import TiLocalMessagePassing
from ..ti_dual_graph.sampler import get_directed_edge_sampler
from ..ti_dual_graph.ti_global_graph import TiGlobalGraphBuilder
from ..ti_dual_graph.ti_local_graph import TiLocalGraphBuilder


class TiGNN():
    def __init__(self, node_features, edge_features, device,
                 edge_memory_config, time_dimension,
                 ti_neighbors_config, edge_neighbors_agg_config,
                 use_ti_local, use_ti_global,
                 ti_local_edge_sampler_config, ti_global_edge_sampler_config, ti_local_mp_config, ti_global_mp_config,
                 embedding_neighbors_config, embedding_module_config,
                 ):
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
        edge_memory_dim = edge_memory_config['edge_memory_dimension'] if edge_memory_config['edge_memory_dimension'] is None else self.edge_features_dim
        self.edge_memory = get_edge_memory(edge_memory_config['memory_init_name'],
                                           n_edges=self.n_edges,
                                           memory_dim=edge_memory_dim,
                                           sources=edge_memory_config['sources'],
                                           destinations=edge_memory_config['destinations'],
                                           node_features=node_features,
                                           fusion_fn_name=edge_memory_config['fusion_fn_name'])
        self.edge_memory_dim = self.edge_memory.memory_dimension

        # use_local/global
        self.use_ti_local = use_ti_local
        self.use_ti_global = use_ti_global

        # graph builders and mailbox
        ti_local_graph_builder = self.init_ti_graph_builder(TiLocalGraphBuilder, ti_local_edge_sampler_config) \
            if use_ti_local else None
        ti_global_graph_builder = self.init_ti_graph_builder(TiGlobalGraphBuilder, ti_global_edge_sampler_config) \
            if use_ti_global else None

        self.mailbox = MailBox(ti_local_graph_builder=ti_local_graph_builder,
                               ti_global_graph_builder=ti_global_graph_builder,
                               neighbor_finder=ti_neighbors_config['neighbor_finder'],
                               ti_n_neighbors=ti_neighbors_config['ti_n_neighbors'])

        # edge_neighbors_agg
        self.edge_memory_neighbors_aggregator = get_edge_neighbors_aggregator(agg_name=edge_neighbors_agg_config['memory_agg_name'],
                                                                              edge_features_dim=self.edge_memory_dim,
                                                                              n_neighbors=edge_neighbors_agg_config['n_neighbors'],
                                                                              device=self.device)
        if edge_neighbors_agg_config['features_agg_name'] == 'same':
            self.edge_features_neighbors_aggregator = self.edge_memory_neighbors_aggregator
        else:
            self.edge_features_neighbors_aggregator = get_edge_neighbors_aggregator(agg_name=edge_neighbors_agg_config['features_agg_name'],
                                                                                    edge_features_dim=self.edge_memory_dim,
                                                                                    n_neighbors=edge_neighbors_agg_config['n_neighbors'],
                                                                                    device=self.device)
        # time encoder
        if time_dimension == 'node':
            self.time_dimension = self.node_features_dim
        elif time_dimension == 'edge':
            self.time_dimension = self.edge_features_dim
        elif time_dimension == 'memory':
            self.time_dimension = self.edge_memory_dim
        else:
            self.time_dimension = time_dimension
        if self.time_dimension > 0:
            self.time_encoder = TimeEncode(dimension=self.time_dimension).to(device)
        else:
            self.time_encoder = None

        # local message passing
        if use_ti_local:
            self.ti_local_mp = TiLocalMessagePassing(node_memory_dim=self.edge_memory_dim,
                                                     node_features_dim=self.edge_features_dim,
                                                     edge_features_dim=self.node_features_dim,
                                                     time_encoding_dim=self.time_dimension,
                                                     device=self.device,
                                                     use_causal=ti_local_mp_config['use_causal'],
                                                     causal_config=ti_local_mp_config['causal_config'],
                                                     use_conseq=ti_local_mp_config['use_conseq'],
                                                     conseq_config=ti_local_mp_config['conseq_config'],
                                                     fusion_fn_name=ti_local_mp_config['fusion_fn_name'],
                                                     is_together=ti_local_mp_config['is_together']).to(device)
        else:
            self.ti_local_mp = None

        # global message passing
        if use_ti_global:
            self.ti_global_mp = TiLocalMessagePassing(node_memory_dim=self.edge_memory_dim,
                                                     node_features_dim=self.edge_features_dim,
                                                     edge_features_dim=self.node_features_dim,
                                                     time_encoding_dim=self.time_dimension,
                                                     device=self.device,
                                                     use_causal=ti_global_mp_config['use_causal'],
                                                     causal_config=ti_global_mp_config['causal_config'],
                                                     use_conseq=ti_global_mp_config['use_conseq'],
                                                     conseq_config=ti_global_mp_config['conseq_config'],
                                                     fusion_fn_name=ti_global_mp_config['fusion_fn_name'],
                                                     is_together=ti_global_mp_config['is_together']).to(device)
        else:
            self.ti_global_mp = None

        # embedding module



        # init embedding module
        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     node_features=self.node_raw_features,
                                                     edge_features=self.edge_raw_features,
                                                     memory=self.memory,
                                                     neighbor_finder=embedding_neighbor_finder,
                                                     time_encoder=self.time_encoder_embedding,
                                                     n_layers=embedding_n_layers,
                                                     n_node_features=self.n_node_features,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=self.n_node_features,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=embedding_n_heads, dropout=embedding_dropout,
                                                     use_memory=True,
                                                     n_neighbors=self.embedding_n_neighbors)


        # MLP to compute probability on an edge given two node embeddings
        self.affinity_score = MergeNodesLayer(self.node_features_dim,
                                              self.node_features_dim,
                                              self.node_features_dim,
                                              1).to(self.device)

    def init_ti_graph_builder(self, ti_graph_builder_class, edge_sampler_config):
        edge_sampler = get_directed_edge_sampler(sampler_name=edge_sampler_config['sampler_name'],
                                                       sampling_fn_name=edge_sampler_config['sampling_fn_name'],
                                                       n_sample=edge_sampler_config['n_sample'],
                                                       min_n_sample=edge_sampler_config['min_n_sample'],
                                                       max_n_sample=edge_sampler_config['max_n_sample'])
        return ti_graph_builder_class(edge_sampler)

    def init_edge_time_encoder(self, edge_time_dimension):
        if edge_time_dimension is None:
            return None
        if isinstance(edge_time_dimension, str):
            if edge_time_dimension == 'node_features':
                return self.node_features_dim
            if edge_time_dimension == 'edge_features':
                return self.edge_features_dim
        return edge_time_dimension

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                   edge_idxs, mode='eval', n_neighbors=20):
        n_samples = len(source_nodes)
        source_node_embedding, destination_node_embedding, negative_node_embedding, updated_edge_memory, edge_memory_ids = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, mode, n_neighbors)

        score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                    torch.cat([destination_node_embedding,
                                               negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]
        return pos_score.sigmoid(), neg_score.sigmoid(), updated_edge_memory, edge_memory_ids

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes,
                                    edge_times, edge_idxs, mode, n_neighbors=20):
        # Update memory for all nodes with interactions stored in mailbox
        updated_memory, edge_ids = self.get_updated_memory()
        n_samples = len(source_nodes)
        nodes = np.concatenate(
            [source_nodes, destination_nodes, negative_nodes]).astype(int)
        # positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])

        # Compute the embeddings using the embedding module
        node_embedding = self.embedding_module.compute_embedding(memory=node_memory,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.embedding_n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=None)
        # update mailbox
        self.mailbox.clear()
        self.mailbox.add_batch(source_nodes, destination_nodes, edge_times, edge_idxs)

        # get final representation
        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples:]
        if mode == 'train':
            return source_node_embedding, destination_node_embedding, negative_node_embedding, None, None

        return source_node_embedding, destination_node_embedding, negative_node_embedding, None, None

    def get_updated_memory(self):
        if not self.mailbox.is_empty():
            start_interval_time = self.mailbox.get_start_interval_time()
            ti_dual_adj_list, local_ti_dual_graph, global_ti_dual_graph, neighbors_edge_idxs = \
                self.mailbox.get_mail_graphs(start_interval_time)

            dual_edge_features = self.node_features[ti_dual_adj_list.edge_features_ids]

            real_edge_ids = list(range(neighbors_edge_idxs.shape[0], len(ti_dual_adj_list.edge_features_ids)))


            dual_node_memory = self.get_dual_node_vectors(edge_vectors=self.edge_memory.memory,
                                                          edge_ids=real_edge_ids,
                                                          extra_aggregator=self.edge_memory_neighbors_aggregator,
                                                          neighbors_edge_idxs=neighbors_edge_idxs)
            dual_node_features = self.get_dual_node_vectors(edge_vectors=self.edge_features,
                                                            edge_ids=real_edge_ids,
                                                            extra_aggregator=self.edge_features_neighbors_aggregator,
                                                            neighbors_edge_idxs=neighbors_edge_idxs)
            if self.use_ti_local:
                local_time_encoding = self.encode_time(local_ti_dual_graph) if self.use_ti_local else None
                dual_node_memory = self.ti_local_mp(ti_graph=local_ti_dual_graph,
                                                     node_memory=dual_node_memory,
                                                     node_features=dual_node_features,
                                                     edge_features=dual_edge_features,
                                                     time_encoding=local_time_encoding)

            if self.use_ti_global:
                global_time_encoding = self.encode_time(global_ti_dual_graph) if self.use_ti_global else None
                dual_node_memory = self.ti_global_mp(ti_graph=global_ti_dual_graph,
                                                      node_memory=dual_node_memory,
                                                      node_features=dual_node_features,
                                                      edge_features=dual_edge_features,
                                                      time_encoding=global_time_encoding)
            return dual_node_memory[real_edge_ids], ti_dual_adj_list.node_features_ids[real_edge_ids]
        return None, []

    def get_dual_node_vectors(self, edge_vectors, edge_ids, extra_aggregator, neighbors_edge_idxs):
        real_dual_node_vectors = edge_vectors[edge_ids].to(self.device)
        extra_dual_node_vectors = extra_aggregator(edge_vectors, neighbors_edge_idxs)
        return torch.cat([extra_dual_node_vectors, real_dual_node_vectors], dim=0)

    def encode_time(self, ti_dual_graph):
        pass


    def set_neighbor_finder(self, neighbor_finder):
        self.mailbox.ti_neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
