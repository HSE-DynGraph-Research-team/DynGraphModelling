import numpy as np

from models.TI_GNN.ti_dual_graph.ti_graph_builder import TiDualAdjListBuilder


class MailBox:

    def __init__(self, ti_local_graph_builder, ti_global_graph_builder, neighbor_finder, ti_n_neighbors):
        # super().__init__()
        self.use_local_ti_graph = ti_local_graph_builder is not None
        self.ti_local_graph_builder = ti_local_graph_builder

        self.use_global_ti_graph = ti_global_graph_builder is not None
        self.ti_global_graph_builder = ti_global_graph_builder

        self.neighbor_finder = neighbor_finder
        self.ti_n_neighbors = ti_n_neighbors
        self.raw_interactions = None

    def clear(self):
        self.raw_interactions = None

    def add_batch(self, source_nodes, destination_nodes, edge_times, edge_idxs):
        self.raw_interactions = RawInteractions(source_nodes, destination_nodes, edge_times, edge_idxs)

    def get_start_interval_time(self):
        return self.raw_interactions.edge_times.min() if self.raw_interactions is not None else None

    def _get_mail_full_graph(self, start_interval_time):
        final_source_nodes = self.raw_interactions.source_nodes
        final_destination_nodes = self.raw_interactions.destination_nodes
        final_edge_times = self.raw_interactions.edge_times
        final_edge_idxs = self.raw_interactions.edge_idxs

        node_idxs = np.unique(np.concatenate([final_source_nodes, final_destination_nodes]))
        neighbors_edge_idxs = None
        if self.ti_n_neighbors > 0:
            neighbors_node_idxs, neighbors_edge_idxs, neighbors_edge_times =\
                self.neighbor_finder.get_temporal_neighbor(node_idxs,
                                                           [start_interval_time] * len(node_idxs),
                                                           n_neighbors=self.ti_n_neighbors)
            neighbors_nzero_ids = np.where(np.sum(neighbors_node_idxs, axis=1) > 0)[0]
            if len(neighbors_nzero_ids) > 0:
                neighbors_node_idxs = neighbors_node_idxs[neighbors_nzero_ids, :]
                neighbors_edge_idxs = neighbors_edge_idxs[neighbors_nzero_ids, :]
                neighbors_edge_times = neighbors_edge_times[neighbors_nzero_ids, :]
                nodes_with_neighbors = node_idxs[neighbors_nzero_ids]

                max_ids = np.argmax(neighbors_edge_times, axis=1)
                all_nodes_ids_lst = list(range(len(max_ids)))
                neighbors_edge_times = neighbors_edge_times[all_nodes_ids_lst, max_ids]
                edge_idxs = neighbors_edge_idxs[all_nodes_ids_lst, max_ids]
                neighbors_node_idxs = neighbors_node_idxs[all_nodes_ids_lst, max_ids]

                final_source_nodes = np.concatenate([nodes_with_neighbors, final_source_nodes])
                final_destination_nodes = np.concatenate([neighbors_node_idxs, final_destination_nodes])
                final_edge_times = np.concatenate([neighbors_edge_times, final_edge_times])
                final_edge_idxs = np.concatenate([edge_idxs, final_edge_idxs])
        return TiDualAdjListBuilder(
            sources=final_source_nodes,
            destinations=final_destination_nodes,
            timestamps=final_edge_times,
            node_idxs=node_idxs,
            edge_idxs=final_edge_idxs).build(), neighbors_edge_idxs  # neighbors_edge_idxs are always less than edge_idsx in batch

    def is_empty(self):
        return self.raw_interactions is None

    def get_mail_graphs(self, start_interval_time):
        ti_dual_adj_list, neighbors_edge_idxs = self._get_mail_full_graph(start_interval_time)
        local_ti_dual_graph = None if self.use_local_ti_graph is None \
            else self.ti_local_graph_builder.build(ti_dual_adj_list)
        global_ti_dual_graph = None if self.use_global_ti_graph is None \
            else self.ti_global_graph_builder.build(ti_dual_adj_list)
        return ti_dual_adj_list, local_ti_dual_graph, global_ti_dual_graph, neighbors_edge_idxs


class RawInteractions:
    def __init__(self, source_nodes, destination_nodes, edge_times, edge_idxs):
        self.source_nodes = source_nodes
        self.destination_nodes = destination_nodes
        self.edge_times = edge_times
        self.edge_idxs = edge_idxs
