from collections import defaultdict
from itertools import combinations
import dgl
import numpy as np
import torch



# def transform_nodes_to_start(nodes):
#     sorted_nodes = np.sort(nodes)
#     graph_nodes_to_real_nodes = {i: node for i, node in enumerate(sorted_nodes)}
#     return graph_to_real, np.array([graph_node_to_real_id[xi] for xi in destinations])



class NodeEdgeIncidence:
    def __init__(self, node_to_edges, node_cnt, edge_cnt):
        self.indices, self.values = self.init_index_values(node_to_edges)
        self.node_cnt = node_cnt
        self.edge_cnt = edge_cnt
        self.coo_tensor = torch.sparse_coo_tensor(self.indices, self.values, size=(node_cnt, edge_cnt))

    def init_index_values(self, node_to_edges):
        rows, cols = [], []
        for node, edges in node_to_edges.items():
            rows.extend([node] * len(edges))
            cols.extend(edges)
        values = torch.tensor([1] * len(rows), dtype=torch.int8)
        indices = torch.LongTensor([rows, cols])
        return indices, values



def get_neighbor_finder(data, uniform, max_node_idx=None):
    max_node_idx = max(data.edge_table[:,2].max(), data.edge_table[:,3].max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(int(max_node_idx) + 1)]
    for edge_idx, timestamp, source, destination in zip(
        data.edge_table[:,0],
        data.edge_table[:,1],
        data.edge_table[:,2],
        data.edge_table[:,3],
            ):
        try:
            adj_list[int(source)].append((int(destination), edge_idx, timestamp))
            adj_list[int(destination)].append((int(source), edge_idx, timestamp))
        except:
            print(edge_idx, timestamp, source, destination)
            raise
    return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = -1
          edge_times[i, n_neighbors - len(source_edge_times):] = -1
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = -1

    return neighbors, edge_idxs, edge_times



def build_conseq_in_adj_list_from_causal(causal_in_adj_list):
    conseq_in_adj_list = defaultdict(list)
    for node in causal_in_adj_list.keys():
        for adj_node, adj_edge in causal_in_adj_list[node]:
            conseq_in_adj_list[adj_node].append((node, adj_edge))
    return conseq_in_adj_list