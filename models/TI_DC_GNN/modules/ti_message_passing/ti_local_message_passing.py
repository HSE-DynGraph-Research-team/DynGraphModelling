from collections import defaultdict

import torch.nn as nn

from models.TI_DC_GNN.graph.ti_local_graph import LocalDualGraph
from models.TI_DC_GNN.modules.ti_message_passing.dual_directed_mp_layer import LocalDualDirectedMessagePassingLayer
from models.TI_DC_GNN.functions.fusion_fn import get_fusion_fn
from models.TI_DC_GNN.modules.ti_message_passing.edge_memory_update_layer import EdgeMemoryUpdateLayer
from models.TI_DC_GNN.utils.utils import FeatureProperties, MemoryProperties


class TiLocalMessagePassing(nn.Module):

    def __init__(self, dimension_config, device,
                 causal_config, conseq_config, n_causal_steps, fusion_fn_config,
                 edge_memory_update_layer_config):
        super().__init__()
        self.device = device
        self.graph_causal_mp = LocalDualDirectedMessagePassingLayer(dimension_config=dimension_config,
                                                                    device=device,
                                                                    source_memory_readout_config=causal_config['source_memory_readout_config'],
                                                                    message_builder_config=causal_config['message_builder_config'],
                                                                    dest_memory_readout_config=causal_config['dest_memory_readout_config'],
                                                                    message_aggregator_config=causal_config['message_aggregator_config'],
                                                                    memory_updater_config=causal_config['node_memory_updater_config'],
                                                                    dest_memory_write_in_fn_config=causal_config['dest_memory_write_in_fn_config'],)

        self.use_conseq = conseq_config['use_conseq']
        if self.use_conseq:
            if conseq_config is None:
                self.graph_conseq_mp = self.graph_causal_mp
            else:
                self.graph_conseq_mp = LocalDualDirectedMessagePassingLayer(dimension_config,
                                                                            device=device,
                                                                            source_memory_readout_config=conseq_config['source_memory_readout_config'],
                                                                            message_builder_config=conseq_config['message_builder_config'],
                                                                            dest_memory_readout_config=conseq_config['dest_memory_readout_config'],
                                                                            message_aggregator_config=conseq_config['message_aggregator_config'],
                                                                            memory_updater_config=conseq_config['memory_updater_config'],
                                                                            dest_memory_write_in_fn_config=conseq_config['dest_memory_write_in_fn_config'],)
                self.fusion_fn = get_fusion_fn(fn_name=fusion_fn_config['fn_name'],
                                               act_fn_name=fusion_fn_config['act_fn_name'],
                                               input_dim=self.graph_causal_mp.get_output_dim(),
                                               hidden_dim=self.graph_conseq_mp.get_output_dim())

        self.n_causal_steps = n_causal_steps
        self.do_edge_mem_update = edge_memory_update_layer_config['do_edge_mem_update']
        if self.do_edge_mem_update:
            self.edge_memory_update_layer = EdgeMemoryUpdateLayer(dimension_config=dimension_config,
                                                                  node_memory_readout_config=edge_memory_update_layer_config['node_memory_readout_config'],
                                                                  dest_edge_memory_readout_config=edge_memory_update_layer_config['dest_edge_memory_readout_config'],
                                                                  message_builder_config=edge_memory_update_layer_config['message_builder_config'],
                                                                  message_aggregator_config=edge_memory_update_layer_config['message_aggregator_config'],
                                                                  memory_updater_config=edge_memory_update_layer_config['memory_updater_config'],
                                                                  memory_write_in_config=edge_memory_update_layer_config['memory_write_in_config'],
                                                                  source_memory_readout_fn=self.graph_causal_mp.source_memory_readout_fn,
                                                                  edge_memory_readout_fn=self.graph_causal_mp.edge_memory_readout_fn,
                                                                  message_builder=self.graph_causal_mp.message_builder,
                                                                  message_aggregator=self.graph_causal_mp.message_aggregator,
                                                                  memory_updater=self.graph_causal_mp.memory_updater,
                                                                  memory_write_in_fn=self.graph_causal_mp.memory_write_in_fn)

            self.edge_memory_update_period_type = edge_memory_update_layer_config['period_type']  # every, after_both, after_conseq

    def forward_causal(self, local_dual_graph: LocalDualGraph, feature_properties: FeatureProperties,
                       memory_properties: MemoryProperties,
                       time_encoding, layer_num):
        dest_node_ids = local_dual_graph.layer_to_node_ids[layer_num]
        grouped_source_ids, grouped_edge_ids = local_dual_graph.get_nodes_predecessors(dest_node_ids)
        dest_memory, dest_node_ids_torch = self.graph_causal_mp(grouped_source_ids=grouped_source_ids,
                                                    grouped_edge_ids=grouped_source_ids,
                                                    dest_node_ids=grouped_source_ids,
                                                    feature_properties=feature_properties,
                                                    memory_properties=memory_properties,
                                                    time_encoding=time_encoding)
        return dest_memory, dest_node_ids_torch
        if self.do_edge_mem_update and self.edge_memory_update_period_type == 'every':
            node_edge_properties = self.apply_edge_memory_update([dest_node_ids, grouped_source_ids, grouped_edge_ids],
                                                                 node_edge_properties, time_encoding)
        if self.use_conseq:
            conseq_dest_node_ids, conseq_grouped_source_ids, conseq_grouped_edge_ids =\
                local_dual_graph.get_conseq_node_predecessors(dest_node_ids, grouped_source_ids, grouped_edge_ids)
            conseq_list_node_edge_ids.append(
                (conseq_dest_node_ids, conseq_grouped_source_ids, conseq_grouped_edge_ids))
            if self.n_causal_steps is not None and i % self.n_causal_steps == 0:
                if self.do_edge_mem_update and self.edge_memory_update_period_type == 'after_both':
                    node_edge_properties = self.apply_edge_memory_update(conseq_list_node_edge_ids,
                                                                         node_edge_properties,
                                                                         time_encoding)
                causal_node_memory = node_edge_properties.node_memory
                node_edge_properties = self.apply_graph_conseq_mp(conseq_list_node_edge_ids,
                                                                  node_edge_properties,
                                                                  time_encoding)
                node_edge_properties.node_memory = self.appply_fusion_fn(causal_node_memory, node_edge_properties.node_memory)
                if self.edge_memory_update_layer is not None and \
                        self.edge_memory_update_period_type in ['after_both', 'after_conseq']:
                    node_edge_properties = self.apply_edge_memory_update(conseq_list_node_edge_ids,
                                                                         node_edge_properties,
                                                                         time_encoding)
                conseq_list_node_edge_ids.clear()

        if self.edge_memory_update_layer is not None:
            node_edge_properties = self.apply_edge_memory_update(conseq_list_node_edge_ids,
                                                                 node_edge_properties=node_edge_properties,
                                                                 time_encoding=time_encoding)
        return node_edge_properties

    def forward_conseq(self, local_dual_graph: LocalDualGraph, node_edge_properties: NodeEdgeProperties, time_encoding,
                       causal_dest_node_ids, causal_grouped_source_ids, causal_grouped_edge_ids):
        conseq_dest_node_ids, conseq_grouped_source_ids, conseq_grouped_edge_ids =\
            local_dual_graph.get_conseq_node_predecessors(dest_node_ids, grouped_source_ids, grouped_edge_ids)
        conseq_list_node_edge_ids.append(
            (conseq_dest_node_ids, conseq_grouped_source_ids, conseq_grouped_edge_ids))
        if self.n_causal_steps is not None and i % self.n_causal_steps == 0:
            if self.do_edge_mem_update and self.edge_memory_update_period_type == 'after_both':
                node_edge_properties = self.apply_edge_memory_update(conseq_list_node_edge_ids,
                                                                     node_edge_properties,
                                                                     time_encoding)
            causal_node_memory = node_edge_properties.node_memory
            node_edge_properties = self.apply_graph_conseq_mp(conseq_list_node_edge_ids,
                                                              node_edge_properties,
                                                              time_encoding)
            node_edge_properties.node_memory = self.appply_fusion_fn(causal_node_memory, node_edge_properties.node_memory)
            if self.edge_memory_update_layer is not None and \
                    self.edge_memory_update_period_type in ['after_both', 'after_conseq']:
                node_edge_properties = self.apply_edge_memory_update(conseq_list_node_edge_ids,
                                                                     node_edge_properties,
                                                                     time_encoding)
            conseq_list_node_edge_ids.clear()

        if self.use_conseq:
            conseq_dest_node_ids, conseq_grouped_source_ids, conseq_grouped_edge_ids =\
                local_dual_graph.get_conseq_node_predecessors(dest_node_ids, grouped_source_ids, grouped_edge_ids)
            conseq_list_node_edge_ids.append(
                (conseq_dest_node_ids, conseq_grouped_source_ids, conseq_grouped_edge_ids))
            if self.n_causal_steps is not None and i % self.n_causal_steps == 0:
                if self.do_edge_mem_update and self.edge_memory_update_period_type == 'after_both':
                    node_edge_properties = self.apply_edge_memory_update(conseq_list_node_edge_ids,
                                                                         node_edge_properties,
                                                                         time_encoding)
                causal_node_memory = node_edge_properties.node_memory
                node_edge_properties = self.apply_graph_conseq_mp(conseq_list_node_edge_ids,
                                                                  node_edge_properties,
                                                                  time_encoding)
                node_edge_properties.node_memory = self.appply_fusion_fn(causal_node_memory, node_edge_properties.node_memory)
                if self.edge_memory_update_layer is not None and \
                        self.edge_memory_update_period_type in ['after_both', 'after_conseq']:
                    node_edge_properties = self.apply_edge_memory_update(conseq_list_node_edge_ids,
                                                                         node_edge_properties,
                                                                         time_encoding)
                conseq_list_node_edge_ids.clear()

        if self.edge_memory_update_layer is not None:
            node_edge_properties = self.apply_edge_memory_update(conseq_list_node_edge_ids,
                                                                 node_edge_properties=node_edge_properties,
                                                                 time_encoding=time_encoding)



    def apply_edge_memory_update(self, list_node_edge_ids, node_edge_properties, time_encoding):
        edge_ids, grouped_node_ids = group_node_edges_for_update(list_node_edge_ids)
        node_edge_properties = self.edge_memory_update_layer(edge_ids=edge_ids,
                                                             grouped_node_ids=grouped_node_ids,
                                                             node_edge_properties=node_edge_properties,
                                                             time_encoding=time_encoding)
        return node_edge_properties

    def apply_graph_conseq_mp(self, conseq_list_node_edge_ids, node_edge_properties, time_encoding):
        for conseq_dest_node_ids, conseq_grouped_source_ids, conseq_grouped_edge_ids in conseq_list_node_edge_ids[::-1]:
            node_edge_properties = self.graph_conseq_mp(conseq_grouped_source_ids,
                                                        conseq_grouped_edge_ids,
                                                        conseq_dest_node_ids,
                                                        node_edge_properties,
                                                        time_encoding)
            if self.edge_memory_update_layer is not None and self.edge_memory_update_period_type == 'every':
                node_edge_properties = self.apply_edge_memory_update(conseq_dest_node_ids,
                                                                     conseq_grouped_source_ids,
                                                                     conseq_grouped_edge_ids)
        return node_edge_properties


def group_node_edges_for_update(list_node_edge_ids):
    edge_id_to_node_ids = defaultdict(set)
    for grouped_source_ids, grouped_edge_ids, dest_node_ids in list_node_edge_ids:
        for i in range(len(dest_node_ids)):
            for edge_id, node_id in zip(grouped_edge_ids[i], grouped_source_ids[i]):
                edge_id_to_node_ids[edge_id].add(node_id)
                edge_id_to_node_ids[edge_id].add(dest_node_ids[i])
    return list(edge_id_to_node_ids.keys()), [list(node_ids) for node_ids in edge_id_to_node_ids.values()]
