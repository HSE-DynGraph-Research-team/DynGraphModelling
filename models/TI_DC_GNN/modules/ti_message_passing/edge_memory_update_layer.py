import torch.nn as nn

from models.TI_DC_GNN.modules.ti_message_passing.memory_read_out import get_memory_readout
from models.TI_DC_GNN.modules.ti_message_passing.memory_updater import get_memory_updater
from models.TI_DC_GNN.modules.ti_message_passing.memory_write_in import get_memory_write_in
from models.TI_DC_GNN.modules.ti_message_passing.message_aggregator import get_message_aggregator
from models.TI_DC_GNN.modules.ti_message_passing.message_builder import get_message_builder
from models.TI_DC_GNN.utils.utils import NodeEdgeProperties


class EdgeMemoryUpdateLayer(nn.Module):
    def __init__(self, dimension_config, node_memory_readout_config, dest_edge_memory_readout_config,
                 message_builder_config, message_aggregator_config,
                 memory_updater_config, memory_write_in_config,
                 source_memory_readout_fn, edge_memory_readout_fn,
                 message_builder, message_aggregator, memory_updater, memory_write_in_fn):
        super().__init__()
        if node_memory_readout_config['share_weights']:
            self.source_memory_readout_fn = source_memory_readout_fn
        else:
            self.source_memory_readout_fn = get_memory_readout(memory_dim=dimension_config.node_memory_dim,
                                                               features_dim=dimension_config.node_features_dim,
                                                               fusion_fn_config=node_memory_readout_config['fusion_fn_config'],
                                                               transform_memory_fn_config=node_memory_readout_config['transform_memory_fn_config'],
                                                               transform_feats_fn_config=node_memory_readout_config['transform_feats_fn_config'])

        if dest_edge_memory_readout_config['share_weights']:
            self.edge_memory_readout_fn = edge_memory_readout_fn
        else:
            self.edge_memory_readout_fn = get_memory_readout(memory_dim=dimension_config.node_memory_dim,
                                                             features_dim=dimension_config.node_features_dim,
                                                             fusion_fn_config=dest_edge_memory_readout_config['fusion_fn_config'],
                                                             transform_memory_fn_config=dest_edge_memory_readout_config['transform_memory_fn_config'],
                                                             transform_feats_fn_config=dest_edge_memory_readout_config['transform_feats_fn_config'])

        if message_builder_config['share_weights']:
            self.message_builder = message_builder
        else:
            self.message_builder = get_message_builder(node_input=message_builder_config['node_input'],
                                                       node_features_dim=self.source_memory_readout_fn.get_output_dim(),
                                                       edge_features_dim=self.edge_memory_readout_fn.get_output_dim(),
                                                       time_encoding_dim=dimension_config.time_encoding_dim,
                                                       node_time_fusion_fn_config=message_builder_config['node_time_fusion_fn_config'],
                                                       edge_fusion_fn_config=message_builder_config['edge_fusion_fn_config'])
        if message_aggregator_config['share_weights']:
            self.message_aggregator = message_aggregator
        else:
            self.message_aggregator = get_message_aggregator(target_node_features_dim=self.dest_memory_readout_fn.get_output_dim(),
                                                             message_dim=self.message_builder.get_output_dim(),
                                                             target_time_fusion_config=message_aggregator_config['target_time_fusion_config'],
                                                             aggregation_config=message_aggregator_config['aggregation_config'])
        if memory_updater_config['share_weights']:
            self.memory_updater = memory_updater
        else:
            self.memory_updater = get_memory_updater(memory_dim=dimension_config.node_memory_dim,
                                                     message_dim=self.message_aggregator.get_output_dim(),
                                                     fusion_fn_config=memory_updater_config['fusion_fn_config'])
        if memory_write_in_config['share_weights']:
            self.memory_write_in_fn = memory_write_in_fn
        else:
            self.memory_write_in_fn = get_memory_write_in(old_node_memory_dim=dimension_config.node_memory_dim,
                                                          updated_memory_dim=self.memory_updater.get_output_dim(),
                                                          fusion_fn_config=memory_write_in_config['fusion_fn_config'])

    def get_output_dim(self):
        return self.memory_write_in_fn.get_output_dim()

    def forward(self, grouped_source_ids, grouped_edge_ids,  dest_node_ids, node_edge_properties: NodeEdgeProperties, time_encoding):
        pass
