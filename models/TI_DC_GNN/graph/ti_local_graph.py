import random

from models.TI_DC_GNN.graph.causal_adj_list import CausalInAdjList


class LocalDualGraph:
    def __init__(self, causal_in_adj_list: CausalInAdjList, layer_to_node_ids, predecessor_samplers, sample_once):
        self.predecessor_samplers = predecessor_samplers
        self.layer_to_node_ids = layer_to_node_ids
        self.causal_in_adj_list = causal_in_adj_list
        self.sample_once = sample_once
        if sample_once:
            self.causal_in_adj_list.node_to_grouped_predecessors =\
                self.sample_predecessors(self.causal_in_adj_list.node_to_grouped_predecessors)

    def sample_predecessors(self, node_to_grouped_predecessors):
        if len(self.predecessor_samplers) == 1:
            predecessor_sampler = self.predecessor_samplers[0]
        else:
            predecessor_sampler = self.predecessor_samplers[random.randint(0, len(self.predecessor_samplers))]
        return predecessor_sampler.sample(node_to_grouped_predecessors)

    def get_nodes_predecessors(self, node_ids):
        node_to_grouped_predessors = self.causal_in_adj_list.node_to_grouped_predecessors
        if not self.sample_once:
            node_to_grouped_predessors = \
                {node: self.causal_in_adj_list.node_to_grouped_predecessors[node] for node in node_ids}
            node_to_grouped_predessors = self.sample_predecessors(node_to_grouped_predessors)
        grouped_sources, grouped_edges = [], []
        for node in node_ids:
            cur_sources, cur_edges = [], []
            for edge in node_to_grouped_predessors[node]:
                cur_sources.extend(node_to_grouped_predessors[node][edge])
                cur_edges.extend([edge] * len(node_to_grouped_predessors[node][edge]))
            grouped_sources.append(cur_sources)
            grouped_edges.append(cur_edges)
        return grouped_sources, grouped_edges


class LocalGraphBuilder:
    def __init__(self, predecessor_samplers, sample_once: bool):
        self.predecessor_samplers = predecessor_samplers
        self.sample_once = sample_once

    def build(self, causal_in_adj_list):
        causal_node_layers = LocalGraphBuilder.build_node_layers(causal_in_adj_list)
        return LocalDualGraph(causal_in_adj_list, causal_node_layers, self.predecessor_samplers, self.sample_once)

    @staticmethod
    def build_node_layers(causal_in_adj_list: CausalInAdjList):
        node_to_layer = {}
        max_layer = 0
        for node_id in causal_in_adj_list.node_idxs:
            if len(causal_in_adj_list.node_to_grouped_predecessors[node_id]) == 0:
                node_to_layer[node_id] = 0
                continue
            in_max_layer = max(
                [node_to_layer[in_node_id] for in_node_id, _ in causal_in_adj_list.node_to_grouped_predecessors[node_id]])
            if in_max_layer + 1 > max_layer:
                max_layer = in_max_layer + 1
            node_to_layer[node_id] = max_layer

        node_layers = [[] for _ in range(max_layer + 1)]
        for node, layer in node_to_layer.items():
            node_layers[layer].append(node)
        return node_layers
