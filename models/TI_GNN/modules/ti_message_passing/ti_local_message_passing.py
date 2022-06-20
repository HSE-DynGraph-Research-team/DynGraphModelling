import torch.nn as nn

from models.TI_GNN.modules.ti_message_passing.dual_directed_mp_layer import LocalDualDirectedMessagePassingLayer
from models.TI_GNN.layers.fusion_fn import get_fusion_fn


class TiLocalMessagePassing(nn.Module):

    def __init__(self, node_memory_dim, node_features_dim, edge_features_dim, time_encoding_dim, device,
                 use_causal, causal_config, use_conseq, conseq_config, fusion_fn_name, is_together):
        super().__init__()
        self.device = device
        self.use_causal = use_causal
        if self.use_causal:
            self.graph_causal_mp = LocalDualDirectedMessagePassingLayer(node_memory_dim=node_memory_dim,
                                                                        node_features_dim=node_features_dim,
                                                                        edge_features_dim=edge_features_dim,
                                                                        time_encoding_dim=time_encoding_dim,
                                                                        device=device,
                                                                        source_memory_readout_config=causal_config['source_memory_readout_config'],
                                                                        message_builder_config=causal_config['message_builder_config'],
                                                                        dest_memory_readout_config=causal_config['dest_memory_readout_config'],
                                                                        message_aggregator_config=causal_config['message_aggregator_config'],
                                                                        memory_updater_config=causal_config['memory_updater_config'],
                                                                        dest_memory_write_in_fn_config=causal_config['dest_memory_write_in_fn_config'],)

        self.use_conseq = use_conseq
        if self.use_conseq:
            if conseq_config is None:
                self.graph_conseq_mp = self.graph_causal_mp
            else:
                self.graph_conseq_mp = LocalDualDirectedMessagePassingLayer(node_memory_dim=node_memory_dim,
                                                                            node_features_dim=node_features_dim,
                                                                            edge_features_dim=edge_features_dim,
                                                                            time_encoding_dim=time_encoding_dim,
                                                                            device=device,
                                                                            source_memory_readout_config=conseq_config['source_memory_readout_config'],
                                                                            message_builder_config=conseq_config['message_builder_config'],
                                                                            dest_memory_readout_config=conseq_config['dest_memory_readout_config'],
                                                                            message_aggregator_config=conseq_config['message_aggregator_config'],
                                                                            memory_updater_config=conseq_config['memory_updater_config'],
                                                                            dest_memory_write_in_fn_config=conseq_config['dest_memory_write_in_fn_config'],)
        self.fusion_fn = get_fusion_fn(fusion_fn_name)
        self.is_together = is_together

    def forward(self, ti_graph, node_memory, node_features, edge_features, time_encoding):
        if self.is_together:
            node_memory = self.forward_together(ti_graph, node_memory, node_features, edge_features, time_encoding)
            return node_memory
        node_memory_causal, node_memory_conseq = self.forward_seq(ti_graph, node_memory, node_features, edge_features, time_encoding)
        return self.fusion_fn(node_memory_causal, node_memory_conseq)

    def forward_together(self, ti_graph, node_memory, node_features, edge_features, time_encoding):
        dual_graph_causal = ti_graph.dual_graph_causal
        dual_graph_conseq = ti_graph.dual_graph_conseq
        n_layers = len(dual_graph_causal.node_layers)
        for i in range(1, n_layers):
            if self.use_causal:
                node_memory = self.graph_causal_mp(dual_graph_causal, dual_graph_causal.node_layers[i], node_memory, node_features, edge_features, time_encoding)
            if self.use_conseq:
                node_memory = self.graph_conseq_mp(dual_graph_conseq, dual_graph_conseq.node_layers[n_layers-i], node_memory, node_features, edge_features, time_encoding)
        return node_memory

    def forward_seq(self, ti_graph, node_memory, node_features, edge_features, time_encoding):
        dual_graph_causal = ti_graph.dual_graph_causal
        n_layers = len(dual_graph_causal.node_layers)
        if self.use_causal:
            for i in range(1, n_layers):
                node_memory = self.graph_causal_mp(dual_graph_causal, dual_graph_causal.node_layers[i], node_memory, node_features, edge_features, time_encoding)
        node_memory_causal = node_memory.clone()
        dual_graph_conseq = ti_graph.dual_graph_conseq
        if self.use_conseq:
            for i in range(1, n_layers):
                node_memory = self.graph_causal_mp(dual_graph_conseq, dual_graph_conseq.node_layers[i], node_memory, node_features, edge_features, time_encoding)
        return node_memory_causal, node_memory
