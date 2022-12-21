from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Predecessors:
    node_ids: list[int]
    selected_node_ids: list[int]


@dataclass
class Ancestors:
    node_ids: list[int]


@dataclass
class TreeNode:
    node_id: int
    edge_id_to_predecessors: dict[int, list[Predecessors]]

    def sample_predecessor_node_ids(self, sampling_config):
        pass
        # return grouped_node_ids, grouped_edge_ids  # [[1,2], [1,3,4]]


@dataclass
class DualMPTree:
    node_id_to_tree_node: dict[int, TreeNode]
    layer_to_node_ids: dict[int, list[int]]
    sampling_config: dict

    def get_nodes_predecessors(self, node_ids: list[int]):
        return zip(*[self.node_id_to_tree_node[node_id].sample_predecessor_node_ids(self.sampling_config)
                     for node_id in node_ids])

    def group_node_edges_for_conseq(self, dest_node_ids, grouped_source_ids, grouped_edge_ids):
        node_id_to_ancestor_nodes = defaultdict(list)
        node_id_to_ancestor_edges = defaultdict(list)
        for i in range(len(dest_node_ids)):
            for edge_id, node_id in zip(grouped_edge_ids[i], grouped_source_ids[i]):
                node_id_to_ancestor_nodes[node_id].append(dest_node_ids[i])
                node_id_to_ancestor_edges[node_id].append(edge_id)
        res_node_ids = list(node_id_to_ancestor_nodes.keys())
        grouped_node_ids = [list(node_ids) for node_ids in node_id_to_ancestor_nodes.values()]
        grouped_edge_ids = [list(edge_ids) for edge_ids in node_id_to_ancestor_edges.values()]
        return res_node_ids, grouped_node_ids, grouped_edge_ids

    def get_conseq_node_predecessors(self, dest_node_ids, grouped_source_ids, grouped_edge_ids):
        if self.sampling_config['conseq_by_causal']:
            return self.group_node_edges_for_conseq(dest_node_ids, grouped_source_ids, grouped_edge_ids)
        raise NotImplementedError()  # TODO: ситуация рассинка


# for layer in layer_to_node_id:


class DualMPTreeBuilder:
    def __init__(self):
        pass

    def build(self, original_graph):
        node_id_to_tree_node = self.build_tree_nodes(original_graph)
        layer_to_node_ids = self.build_layers(node_id_to_tree_node, original_graph)
        layer_to_node_ids = self.squash_layers(layer_to_node_ids, node_id_to_tree_node)
        return DualMPTree(node_id_to_tree_node, layer_to_node_ids)

    def build_tree_nodes(self, original_graph):
        pass

    def build_layers(self, node_id_to_tree_node, original_graph):
        pass

    def squash_layers(self, layer_to_node_id, node_id_to_tree_node):
        pass



"""
где-то есть
node_memory
edge_memory (?)
node_timestamp
node_feats
edge_feats
"""