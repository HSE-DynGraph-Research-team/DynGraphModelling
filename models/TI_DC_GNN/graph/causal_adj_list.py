from collections import defaultdict


class CausalInAdjList:
    def __init__(self, node_idxs, node_times, node_to_grouped_predecessors):
        self.node_idxs = node_idxs
        self.node_to_grouped_predecessors = node_to_grouped_predecessors
        self.node_times = node_times


class CausalAdjListBuilder:
    def __init__(self, max_node_predecessors):
        self.max_node_predecessors = max_node_predecessors

    def build(self, raw_interactions):
        edge_idxs = raw_interactions.edge_idxs
        edge_times = raw_interactions.edge_times
        node_to_incident_edges = self.build_incidence_list(raw_interactions)
        edge_to_grouped_predecessors = defaultdict(dict)
        for node in edge_to_grouped_predecessors:
            cur_adj_edges = node_to_incident_edges[node]
            for i in range(len(cur_adj_edges)-1, -1, -1):
                cur_edge_id = cur_adj_edges[i]
                start_predecessor_ind = max(i - 1 - self.max_node_predecessors, 0) if self.max_node_predecessors is not None else 0
                edge_to_grouped_predecessors[cur_edge_id][node] = \
                    [cur_adj_edges[j] for j in range(start_predecessor_ind, i-1)]
        return CausalInAdjList(edge_idxs, edge_times, edge_to_grouped_predecessors)

    def build_incidence_list(self, raw_interactions):
        exception_edges = raw_interactions.exception_edges
        node_to_edges = defaultdict(list)
        for source, dest, edge_idx in zip(raw_interactions.source_nodes,
                                          raw_interactions.destination_nodes,
                                          raw_interactions.edge_idxs):
            if edge_idx not in exception_edges:
                node_to_edges[source].append(edge_idx)
            node_to_edges[dest].append(edge_idx)
        return node_to_edges
