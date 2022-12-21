import torch.nn as nn

from models.TI_GNN_old.modules.ti_message_passing.dual_directed_mp_layer import GlobalDualDirectedMessagePassingLayer
from models.TI_GNN_old.layers.fusion_fn import get_fusion_fn


class TiGlobalMessagePassing(nn.Module):

    def __init__(self, node_memory_dim, node_features_dim, edge_features_dim, time_encoding_dim, device,
                 use_causal, causal_config, use_conseq, conseq_config, fusion_fn_name, is_together):
        super().__init__()
        self.use_causal = use_causal
        if self.use_causal:
            self.graph_causal_mp = DualDirectedMessagePassing(node_memory_dim=node_memory_dim,
                                                              node_features_dim=node_features_dim,
                                                              edge_features_dim=edge_features_dim,
                                                              time_encoding_dim=time_encoding_dim,
                                                              device=device,
                                                              source_memory_readout_config=causal_config['source_memory_readout_config'],
                                                              message_builder_config=causal_config['message_builder_config'],
                                                              dest_memory_readout_config=causal_config['dest_memory_readout_config'],
                                                              message_aggregator_config=causal_config['message_aggregator_config'],
                                                              memory_updater_config=causal_config['memory_updater_config'],
                                                              dest_memory_write_in_fn_config=causal_config['dest_memory_write_in_fn_config'],
                                                              n_layers=causal_config['n_layers'],
                                                              share_weights=causal_config['share_weights'])

        self.use_conseq = use_conseq
        if self.use_conseq:
            if conseq_config is None:
                self.graph_conseq_mp = self.graph_causal_mp
            else:
                self.graph_conseq_mp = DualDirectedMessagePassing(node_memory_dim=node_memory_dim,
                                                                  node_features_dim=node_features_dim,
                                                                  edge_features_dim=edge_features_dim,
                                                                  time_encoding_dim=time_encoding_dim,
                                                                  device=device,
                                                                  source_memory_readout_config=conseq_config['source_memory_readout_config'],
                                                                  message_builder_config=conseq_config['message_builder_config'],
                                                                  dest_memory_readout_config=conseq_config['dest_memory_readout_config'],
                                                                  message_aggregator_config=conseq_config['message_aggregator_config'],
                                                                  memory_updater_config=conseq_config['memory_updater_config'],
                                                                  dest_memory_write_in_fn_config=conseq_config['dest_memory_write_in_fn_config'],
                                                                  n_layers=conseq_config['n_layers'],
                                                                  share_weights=conseq_config['share_weights'])
        self.n_layers_mp = causal_config['n_layers'] if self.use_causal else conseq_config['n_layers']
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
        for i in range(self.n_layers_mp):
            if self.use_causal:
                node_memory = self.graph_causal_mp.forward_layer(dual_graph_causal, node_memory, node_features, edge_features, time_encoding, i)
            if self.use_conseq:
                node_memory = self.graph_conseq_mp.forward_layer(dual_graph_conseq, node_memory, node_features, edge_features, time_encoding, i)
        return node_memory

    def forward_seq(self, ti_graph, node_memory, node_features, edge_features, time_encoding):
        dual_graph_causal = ti_graph.dual_graph_causal
        if self.use_causal:
            node_memory = self.graph_causal_mp.forward_layers(dual_graph_causal, node_memory, node_features, edge_features, time_encoding)
        node_memory_causal = node_memory.clone()
        dual_graph_conseq = ti_graph.dual_graph_conseq
        if self.use_conseq:
            node_memory = self.graph_conseq_mp.forward_layers(dual_graph_conseq, node_memory, node_features, edge_features, time_encoding)
        return node_memory_causal, node_memory

class DualDirectedMessagePassing(nn.Module):

    def __init__(self, node_memory_dim, node_features_dim, edge_features_dim, time_encoding_dim, device, n_layers, share_weights,
                 source_memory_readout_config, message_builder_config, dest_memory_readout_config,
                 message_aggregator_config, memory_updater_config, dest_memory_write_in_fn_config):
        super().__init__()
        self.device = device
        self.share_weights = share_weights
        self.n_layers = n_layers
        if share_weights:
            self.dual_directed_mp_layer = GlobalDualDirectedMessagePassingLayer(node_memory_dim=node_memory_dim,
                                                                        node_features_dim=node_features_dim,
                                                                        edge_features_dim=edge_features_dim,
                                                                        time_encoding_dim=time_encoding_dim,
                                                                        device=device,
                                                                        source_memory_readout_config=source_memory_readout_config,
                                                                        message_builder_config=message_builder_config,
                                                                        dest_memory_readout_config=dest_memory_readout_config,
                                                                        message_aggregator_config=message_aggregator_config,
                                                                        memory_updater_config=memory_updater_config,
                                                                        dest_memory_write_in_fn_config=dest_memory_write_in_fn_config)
            self.dual_directed_mp_layers = [self.dual_directed_mp_layer for _ in range(n_layers)]
        else:
            self.dual_directed_mp_layers = nn.ModuleList([GlobalDualDirectedMessagePassingLayer(node_memory_dim=node_memory_dim,
                                                                        node_features_dim=node_features_dim,
                                                                        edge_features_dim=edge_features_dim,
                                                                        time_encoding_dim=time_encoding_dim,
                                                                        device=device,
                                                                        source_memory_readout_config=source_memory_readout_config,
                                                                        message_builder_config=message_builder_config,
                                                                        dest_memory_readout_config=dest_memory_readout_config,
                                                                        message_aggregator_config=message_aggregator_config,
                                                                        memory_updater_config=memory_updater_config,
                                                                        dest_memory_write_in_fn_config=dest_memory_write_in_fn_config)
                                            for _ in range(n_layers)])

    def forward_layers(self, dual_graph, node_memory, node_features, edge_features, time_encoding):
        for layer_num in range(self.n_layers):
            node_memory = self.forward_layer(dual_graph, node_memory, node_features, edge_features, time_encoding, layer_num)
        return node_memory


    def forward_layer(self, dual_graph, node_memory, node_features, edge_features, time_encoding, layer_num):
        return self.dual_directed_mp_layer[layer_num](dual_graph, node_memory, node_features, edge_features, time_encoding)

