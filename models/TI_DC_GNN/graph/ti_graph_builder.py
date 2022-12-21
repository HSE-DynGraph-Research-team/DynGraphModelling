from collections import defaultdict
from itertools import combinations
import numpy as np


class TiDualAdjList:
    def __init__(self, causal_in_adj_list, node_features_ids, edge_features_ids):
        self.causal_in_adj_list = causal_in_adj_list
        self.node_features_ids = node_features_ids
        self.edge_features_ids = edge_features_ids


class TiDualAdjListBuilder:

    def __init__(self, sources, destinations, timestamps, node_idxs, edge_idxs):
        self.old_node_idxs = node_idxs
        self.old_node_to_id = {i:node_idx for i, node_idx in enumerate(node_idxs)}
        self.old_edge_idxs = edge_idxs
        self.adj_list, self.edge_to_id = self.build_adj_list(sources, destinations, timestamps, edge_idxs)

    def build_adj_list(self, sources, destinations, timestamps, edge_idxs):
        def add_to_adj_list(source, destination):
            adj_list[source].append((destination, timestamp))
            edge_to_id[(source, destination, timestamp)] = i

        adj_list = defaultdict(list)
        edge_to_id = {}
        for i, (source, destination, timestamp, edge_idx) in enumerate(zip(sources, destinations, timestamps, edge_idxs)):
            source, destination = self.old_node_to_id[source], self.old_node_to_id[destination]
            add_to_adj_list(source, destination)
            add_to_adj_list(destination, source)
        # print_statistics(adj_list)
        return adj_list, edge_to_id

    def build(self):
        causal_in_adj_list = self.build_causal_in_adj_list()
        causal_in_adj_list = self.sort_adj_list(causal_in_adj_list)
        return TiDualAdjList(causal_in_adj_list,
                             node_features_ids=self.old_edge_idxs,
                             edge_features_ids=self.old_node_idxs)

    def sort_adj_list(self, in_adj_list):
        for node in in_adj_list.keys():
            in_adj_list[node] = sorted(in_adj_list[node], key=lambda x: x[0])
            # timestamps = self.dual_nodes_timestamps[[node_idx for node_idx, _ in node_in_adj_list]]
            # sorted_ids = np.argsort(timestamps)
            # in_adj_list[node] = [node_in_adj_list[i] for i in sorted_ids]
        return in_adj_list

    def build_causal_in_adj_list(self):
        causal_in_adj_list = defaultdict(list)
        for source, destinations_timestamps in self.adj_list.items():
            if len(destinations_timestamps) == 0:
                edge_idx = self.edge_to_id[(source,) + destinations_timestamps[0]]
                causal_in_adj_list[edge_idx] = []
                continue
            for dest_timestamp1, dest_timestamp2 in combinations(destinations_timestamps, 2):
                edge_idx1 = self.edge_to_id[(source,) + dest_timestamp1]
                edge_idx2 = self.edge_to_id[(source,) + dest_timestamp2]
                if dest_timestamp1[1] > dest_timestamp2[1]:
                    edge_idx1, edge_idx2 = edge_idx2, edge_idx1
                elif dest_timestamp1[1] == dest_timestamp2[1]:
                    edge_idx1, edge_idx2 = np.random.permutation([edge_idx1, edge_idx2])
                causal_in_adj_list[edge_idx2].append((edge_idx1, source))
        return causal_in_adj_list


def print_statistics(adj_list):
    node_degrees = []
    for node in adj_list:
        node_degrees.append(len(adj_list[node]))
    print('Max degree:', max(node_degrees))
    print('Min degree:', min(node_degrees))
    print('Avg degree:', sum(node_degrees) / len(node_degrees))