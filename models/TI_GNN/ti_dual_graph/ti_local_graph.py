from models.TI_GNN.utils.graph_utils import build_conseq_in_adj_list_from_causal


class TiLocalGraph:
    def __init__(self, causal_in_adj_list, conseq_in_adj_list, causal_node_layers, conseq_node_layers):
        self.dual_graph_causal = LocalDualGraph(causal_in_adj_list, causal_node_layers)
        self.dual_graph_conseq = LocalDualGraph(conseq_in_adj_list, conseq_node_layers)


class LocalDualGraph:
    def __init__(self, in_adj_list, node_layers):
        self.in_adj_list = in_adj_list
        self.node_layers = node_layers

    def get_in_edges(self, node_ids):
        grouped_source_ids, grouped_edge_ids = tuple(zip(*[self.in_adj_list[node] for node in node_ids]))
        return grouped_source_ids, grouped_edge_ids


class TiLocalGraphBuilder:

    def __init__(self, edge_sampler):
        self.edge_sampler = edge_sampler

    def build(self, ti_dual_adj_list):
        causal_in_adj_list = ti_dual_adj_list.causal_in_adj_list
        causal_in_adj_list = self.edge_sampler.sample(causal_in_adj_list)
        conseq_in_adj_list = build_conseq_in_adj_list_from_causal(causal_in_adj_list)
        causal_node_layers = self.build_node_layers(causal_in_adj_list)
        conseq_node_layers = causal_node_layers[::-1]
        return TiLocalGraph(causal_in_adj_list, conseq_in_adj_list, causal_node_layers, conseq_node_layers)


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
