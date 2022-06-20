import torch
import torch.nn as nn

from models.TI_GNN.modules.ti_message_passing.memory_read_out import get_memory_readout
from models.TI_GNN.modules.ti_message_passing.memory_updater import get_memory_updater
from models.TI_GNN.modules.ti_message_passing.memory_write_in import get_memory_write_in
from models.TI_GNN.modules.ti_message_passing.message_aggregator import get_message_aggregator
from models.TI_GNN.modules.ti_message_passing.message_builder import get_message_builder
from models.TI_GNN.utils.utils import flatten_list


class DualDirectedMessagePassingLayer(nn.Module):

    def __init__(self, node_memory_dim, node_features_dim, edge_features_dim, time_encoding_dim, device,
                 source_memory_readout_config, message_builder_config, dest_memory_readout_config,
                 message_aggregator_config, memory_updater_config, dest_memory_write_in_fn_config):
        super().__init__()
        self.device = device
        self.source_memory_readout_fn = get_memory_readout(memory_readout_name=source_memory_readout_config['name'],
                                                           node_memory_dim=node_memory_dim,
                                                           node_features_dim=node_features_dim,
                                                           fusion_fn_name=source_memory_readout_config[
                                                               'fusion_fn_name'],
                                                           transform_fn_name=source_memory_readout_config[
                                                               'transform_fn_name'],
                                                           act_fn_name=source_memory_readout_config['act_fn_name'])
        self.message_builder = get_message_builder(node_input=message_builder_config['node_input'],
                                                   node_features_dim=self.source_memory_readout_fn.get_output_dim(),
                                                   edge_features_dim=edge_features_dim,
                                                   time_encoding_dim=time_encoding_dim,
                                                   message_cell_name=message_builder_config['cell_name'],
                                                   time_fusion_fn_name=message_builder_config['time_fusion_fn_name'],
                                                   time_input=message_builder_config['time_input'],
                                                   act_fn_name=message_builder_config['act_fn_name'])

        if dest_memory_readout_config is None:
            self.dest_memory_readout_fn = self.source_memory_readout_fn
        else:
            self.dest_memory_readout_fn = get_memory_readout(memory_readout_name=dest_memory_readout_config['name'],
                                                             node_memory_dim=node_memory_dim,
                                                             node_features_dim=node_features_dim,
                                                             fusion_fn_name=dest_memory_readout_config[
                                                                 'fusion_fn_name'],
                                                             transform_fn_name=dest_memory_readout_config[
                                                                 'transform_fn_name'],
                                                             act_fn_name=dest_memory_readout_config['act_fn_name'])

        self.message_aggregator = get_message_aggregator(node_features_dim=self.dest_memory_readout_fn.get_output_dim(),
                                                         message_dim=self.message_builder.get_output_dim(),
                                                         aggregation_fn_name=message_aggregator_config['name'],
                                                         agg_name=message_aggregator_config['agg_name'],
                                                         act_fn_name=message_aggregator_config['act_fn_name'])
        self.memory_updater = get_memory_updater(node_memory_dim=node_memory_dim,
                                                 message_dim=self.message_aggregator.get_output_dim(),
                                                 updater_name=memory_updater_config['name'],
                                                 act_fn_name=memory_updater_config['act_fn_name'])
        self.memory_write_in_fn = get_memory_write_in(memory_write_in_name=dest_memory_write_in_fn_config['name'],
                                                      old_node_memory_dim=node_memory_dim,
                                                      updated_memory_dim=self.memory_updater.get_output_dim(),
                                                      act_fn_name=dest_memory_write_in_fn_config['act_fn_name'])

    def get_output_dim(self):
        return self.memory_write_in_fn.get_output_dim()


class LocalDualDirectedMessagePassingLayer(DualDirectedMessagePassingLayer):
    def forward(self, dual_graph, node_ids, node_memory, node_features, edge_features, time_encoding):
        grouped_source_ids, grouped_edge_ids = dual_graph.get_in_edges(node_ids)
        source_ids_flatten = torch.LongTensor(flatten_list(grouped_source_ids))
        edge_ids_flatten = torch.LongTensor(flatten_list(grouped_edge_ids))

        source_memory = self.source_memory_readout_fn(node_memory[source_ids_flatten], node_features[source_ids_flatten])
        messages_flatten = self.message_builder(source_memory, edge_features[edge_ids_flatten], time_encoding)  # (n * message_num) x message_dim
        grouped_messages = list(torch.split(messages_flatten, [len(group) for group in grouped_source_ids]))  # n x message_num x message_dim

        node_ids_torch = torch.LongTensor(node_ids)
        dest_memory = torch.index_select(node_memory, index=node_ids_torch, dim=0)
        dest_memory = self.dest_memory_readout_fn(dest_memory, node_features[node_ids_torch])

        aggregated_messages = self.message_aggregator(dest_memory, grouped_messages)  # n x message_dim
        dest_memory = self.memory_updater(aggregated_messages, dest_memory)
        node_memory[node_ids_torch] = self.memory_write_in_fn(dest_memory)
        return node_memory


class GlobalDualDirectedMessagePassingLayer(DualDirectedMessagePassingLayer):
    def forward(self, dual_graph, node_memory, node_features, edge_features, time_encoding):
        dual_graph.ndata['memory'] = node_memory
        dual_graph.ndata['features'] = node_features
        dual_graph.edata['features'] = edge_features
        dual_graph.edata['time_encoding'] = time_encoding
        dual_graph.update_all(self.dgl_message_fn, self.dgl_reduce_fn)
        return dual_graph['memory']

    def dgl_message_fn(self, edges):
        source_memory = self.source_memory_readout_fn(edges.src['memory'], edges.src['features'])
        messages = self.message_builder(source_memory, edges.data['features'],  edges.data['time_encoding'])  # (n * message_num) x message_dim
        return {'m': messages}

    def dgl_reduce_fn(self, nodes):
        grouped_messages = nodes.mailbox['m']
        dest_memory = self.dest_memory_readout_fn(nodes.data['memory'], nodes.data['features'])
        aggregated_messages = self.message_aggregator(dest_memory, grouped_messages)  # n x message_dim
        dest_memory = self.memory_updater(aggregated_messages, dest_memory)
        dest_memory = self.memory_write_in_fn(dest_memory)
        return {'memory': dest_memory}
