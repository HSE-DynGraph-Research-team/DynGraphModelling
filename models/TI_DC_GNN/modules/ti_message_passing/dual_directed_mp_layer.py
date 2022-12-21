import torch
import torch.nn as nn

from models.TI_DC_GNN.utils.utils import NodeEdgeProperties, FeatureProperties
from models.TI_DC_GNN.modules.ti_message_passing.memory_read_out import get_memory_readout
from models.TI_DC_GNN.modules.ti_message_passing.memory_updater import get_memory_updater
from models.TI_DC_GNN.modules.ti_message_passing.memory_write_in import get_memory_write_in
from models.TI_DC_GNN.modules.ti_message_passing.message_aggregator import get_message_aggregator
from models.TI_DC_GNN.modules.ti_message_passing.message_builder import get_message_builder
from models.TI_DC_GNN.utils.utils import flatten_list


class DualDirectedMessagePassingLayer(nn.Module):

    def __init__(self, dimension_config, device,
                 source_memory_readout_config, edge_memory_readout_config, message_builder_config, dest_memory_readout_config,
                 message_aggregator_config, memory_updater_config, dest_memory_write_in_fn_config,):
        super().__init__()
        self.device = device
        self.source_memory_readout_fn = get_memory_readout(memory_dim=dimension_config.node_memory_dim,
                                                           features_dim=dimension_config.node_features_dim,
                                                           fusion_fn_config=source_memory_readout_config['fusion_fn_config'],
                                                           transform_memory_fn_config=source_memory_readout_config['transform_memory_fn_config'],
                                                           transform_feats_fn_config=source_memory_readout_config['transform_feats_fn_config'])

        self.edge_memory_readout_fn = get_memory_readout(memory_dim=dimension_config.node_memory_dim,
                                                         features_dim=dimension_config.node_features_dim,
                                                         fusion_fn_config=edge_memory_readout_config['fusion_fn_config'],
                                                         transform_memory_fn_config=edge_memory_readout_config['transform_memory_fn_config'],
                                                         transform_feats_fn_config=edge_memory_readout_config['transform_feats_fn_config'])

        self.message_builder = get_message_builder(node_input=message_builder_config['node_input'],
                                                   node_features_dim=self.source_memory_readout_fn.get_output_dim(),
                                                   edge_features_dim=self.edge_memory_readout_fn.get_output_dim(),
                                                   time_encoding_dim=dimension_config.time_encoding_dim,
                                                   node_time_fusion_fn_config=message_builder_config['node_time_fusion_fn_config'],
                                                   edge_fusion_fn_config=message_builder_config['edge_fusion_fn_config'])
        if dest_memory_readout_config['same_as_source_readout']:
            self.dest_memory_readout_fn = self.source_memory_readout_fn
        else:
            self.dest_memory_readout_fn = get_memory_readout(memory_dim=dimension_config.node_memory_dim,
                                                             features_dim=dimension_config.node_features_dim,
                                                             fusion_fn_config=dest_memory_readout_config['fusion_fn_config'],
                                                             transform_memory_fn_config=dest_memory_readout_config['transform_memory_fn_config'],
                                                             transform_feats_fn_config=dest_memory_readout_config['transform_feats_fn_config'])

        self.message_aggregator = get_message_aggregator(target_node_features_dim=self.dest_memory_readout_fn.get_output_dim(),
                                                         message_dim=self.message_builder.get_output_dim(),
                                                         target_time_fusion_config=message_aggregator_config['target_time_fusion_config'],
                                                         aggregation_config=message_aggregator_config['aggregation_config'])

        self.memory_updater = get_memory_updater(memory_dim=dimension_config.node_memory_dim,
                                                 message_dim=self.message_aggregator.get_output_dim(),
                                                 fusion_fn_config=memory_updater_config['fusion_fn_config'])

        self.memory_write_in_fn = get_memory_write_in(old_node_memory_dim=dimension_config.node_memory_dim,
                                                      updated_memory_dim=self.memory_updater.get_output_dim(),
                                                      fusion_fn_config=dest_memory_write_in_fn_config['fusion_fn_config'])

    def get_output_dim(self):
        return self.memory_write_in_fn.get_output_dim()


class LocalDualDirectedMessagePassingLayer(DualDirectedMessagePassingLayer):
    def forward(self, group_sizes, source_ids_flatten, edge_ids_flatten, dest_node_ids_torch, feature_properties: FeatureProperties,
                memory_properties: NodeEdgeProperties, time_encoding):
        source_ids_flatten = torch.LongTensor(flatten_list(grouped_source_ids))
        edge_ids_flatten = torch.LongTensor(flatten_list(grouped_edge_ids))

        source_memory = self.source_memory_readout_fn(memory_properties.node_memory[source_ids_flatten].to(self.device),
                                                      feature_properties.node_features[source_ids_flatten].to(self.device))
        edge_memory = self.edge_memory_readout_fn(memory_properties.edge_memory[edge_ids_flatten].to(self.device),
                                                      feature_properties.edge_features[edge_ids_flatten].to(self.device))
        messages_flatten = self.message_builder(source_memory,
                                                edge_memory,
                                                time_encoding)  # (n * message_num) x message_dim
        grouped_messages = list(torch.split(messages_flatten, [len(group) for group in grouped_source_ids]))  # n x message_num x message_dim

        # dest_memory = torch.index_select(node_memory, index=dest_node_ids_torch, dim=0)
        dest_memory = self.dest_memory_readout_fn(memory_properties.node_memory[dest_node_ids_torch].to(self.device),
                                                  feature_properties.node_features[dest_node_ids_torch].to(self.device))

        aggregated_messages = self.message_aggregator(dest_memory, grouped_messages)  # n x message_dim
        dest_memory = self.memory_updater(aggregated_messages, dest_memory)
        dest_memory = self.memory_write_in_fn(dest_memory)
        return dest_memory, dest_node_ids_torch


# class GlobalDualDirectedMessagePassingLayer(DualDirectedMessagePassingLayer):
#     def forward(self, dual_graph, node_memory, node_features, edge_features, time_encoding):
#         dual_graph.ndata['memory'] = node_memory
#         dual_graph.ndata['features'] = node_features
#         dual_graph.edata['features'] = edge_features
#         dual_graph.edata['time_encoding'] = time_encoding
#         dual_graph.update_all(self.dgl_message_fn, self.dgl_reduce_fn)
#         return dual_graph['memory']
#
#     def dgl_message_fn(self, edges):
#         source_memory = self.source_memory_readout_fn(edges.src['memory'], edges.src['features'])
#         messages = self.message_builder(source_memory, edges.data['features'],  edges.data['time_encoding'])  # (n * message_num) x message_dim
#         return {'m': messages}
#
#     def dgl_reduce_fn(self, nodes):
#         grouped_messages = nodes.mailbox['m']
#         dest_memory = self.dest_memory_readout_fn(nodes.data['memory'], nodes.data['features'])
#         aggregated_messages = self.message_aggregator(dest_memory, grouped_messages)  # n x message_dim
#         dest_memory = self.memory_updater(aggregated_messages, dest_memory)
#         dest_memory = self.memory_write_in_fn(dest_memory)
#         return {'memory': dest_memory}
