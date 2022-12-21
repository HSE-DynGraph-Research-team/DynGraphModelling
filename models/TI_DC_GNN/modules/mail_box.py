import numpy as np

from models.TI_DC_GNN.graph.causal_adj_list import CausalAdjListBuilder


class MailBox:
    def __init__(self, neighbor_finder, old_neighbors_cnt, causal_adj_builder_config,
                 ti_local_graph_builder_config, ti_global_graph_builder_config):
        self.causal_adj_builder = CausalAdjListBuilder(causal_adj_builder_config['max_predecessors'])
        self.ti_local_graph_builder = LocalGraphBuilder(ti_local_graph_builder_config[''])
        self.neighbor_finder = neighbor_finder
        self.old_neighbors_cnt = old_neighbors_cnt
        self.raw_interactions = None

    def clear(self):
        self.raw_interactions = None

    def add_batch(self, source_nodes, destination_nodes, edge_times, edge_idxs):
        self.raw_interactions = RawInteractions(source_nodes, destination_nodes, edge_times, edge_idxs)

    def get_start_interval_time(self):
        if self.raw_interactions is None:
            return None
        return self.raw_interactions.edge_times[0]  # min time

    def get_end_interval_time(self):
        if self.raw_interactions is None:
            return None
        return self.raw_interactions.edge_times[-1]  # max time

    def add_old_interactions(self, start_interval_time):
        if self.old_neighbors_cnt == 0:
            return self.raw_interactions
        node_idxs = np.unique(np.concatenate(
            [self.raw_interactions.source_nodes, self.raw_interactions.destination_nodes]))
        neighbors_node_idxs, neighbors_edge_idxs, neighbors_edge_times = \
            self.neighbor_finder.get_temporal_neighbor(node_idxs,
                                                       [start_interval_time] * len(node_idxs),
                                                       n_neighbors=self.old_neighbors_cnt)
        is_neighbor_array = neighbors_node_idxs > -1
        source_neigh_rows, source_neigh_cols = np.where(is_neighbor_array)
        source_idxs = neighbors_node_idxs[source_neigh_rows, source_neigh_cols]
        edge_idxs = neighbors_edge_idxs[source_neigh_rows, source_neigh_cols]
        edge_times = neighbors_edge_times[source_neigh_rows, source_neigh_cols]
        destination_idxs = np.repeat(node_idxs, np.sum(is_neighbor_array, axis=1))
        self.raw_interactions.add_before(source_idxs, destination_idxs, edge_times, edge_idxs, set(edge_idxs))

    def is_empty(self):
        return self.raw_interactions is None

    def get_causal_adj_list(self, start_interval_time):
        self.add_old_interactions(start_interval_time)
        causal_in_adj_list = self.causal_adj_builder.build(self.raw_interactions)
        return causal_in_adj_list


class RawInteractions:
    def __init__(self, source_nodes, destination_nodes, edge_times, edge_idxs):
        self.source_nodes = source_nodes
        self.destination_nodes = destination_nodes
        self.edge_times = edge_times
        self.edge_idxs = edge_idxs
        self.exception_edges = set()

    def add_before(self, source_nodes, destination_nodes, edge_times, edge_idxs, exception_edges):
        self.source_nodes = np.concatenate[source_nodes, self.source_nodes]
        self.destination_nodes = np.concatenate[destination_nodes, self.destination_nodes]
        self.edge_times = np.concatenate[edge_times, self.edge_times]
        self.edge_idxs = np.concatenate[edge_idxs, self.edge_idxs]
        self.exception_edges = self.exception_edges.union(exception_edges)
