from collections import defaultdict

import numpy as np


class LocalDualGraph:
    def __init__(self, in_adj_list, node_layers, nodes_predecessors):
        self.in_adj_list = in_adj_list
        self.node_layers = node_layers
        self.nodes = np.array(list(range(len(node_layers))))
        self.nodes_predecessors = nodes_predecessors

    def get_in_edges(self, node_ids):
        grouped_source_ids, grouped_edge_ids = tuple(zip(*[self.in_adj_list[node] for node in node_ids]))
        return grouped_source_ids, grouped_edge_ids


class TiLocalGraphBuilder:

    def __init__(self, edge_samplers, max_node_predecessors):
        self.edge_samplers = edge_samplers
        self.max_node_predecessors = max_node_predecessors

    def build(self, causal_in_adj_list):
        causal_node_layers = self.build_node_layers(causal_in_adj_list)
        node_predecessors = self.build_node_predecessors(causal_node_layers, causal_in_adj_list)
        return LocalDualGraph(causal_in_adj_list, causal_node_layers, node_predecessors)

    def build_node_predecessors(self, causal_node_layers, causal_in_adj_list):
        node_predecessors = defaultdict(list)
        layer_to_nodes = defaultdict(list)
        for node in causal_node_layers:
            layer_to_nodes[causal_node_layers[node]].append(node)
        for layer in range(len(layer_to_nodes.keys())):
            for node in layer_to_nodes[layer]:
                node_predecessors[node].a
        return node_predecessors

    def build_node_layers(self, causal_in_adj_list):
        node_to_layer = {}
        max_layer = 0
        for node_id in range(len(causal_in_adj_list)):
            if len(causal_in_adj_list[node_id]) == 0:
                node_to_layer[node_id] = 0
                continue
            in_max_layer = max([node_to_layer[in_node_id] for in_node_id, _ in causal_in_adj_list[node_id]])
            if in_max_layer + 1 > max_layer:
                max_layer = in_max_layer + 1
            node_to_layer[node_id] = max_layer

        node_layers = [[] for _ in range(max_layer + 1)]
        for node, layer in node_to_layer.items():
            node_layers[layer].append(node)
        return node_layers
