import numpy as np
import torch


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round


class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 40)
        self.fc_2 = torch.nn.Linear(40, 2)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        return self.fc_2(x).sigmoid()


def get_neighbor_finder(data, max_node_idx=None, neighbour_max=None):
    max_node_idx = max(data.edge_table[:, 2].max(), data.edge_table[:, 3].max()) if max_node_idx is None else max_node_idx
    #print(max(data.unique_nodes))
    adj_list = [[] for _ in range(int(max_node_idx)+1)]
    adj_time = [[] for _ in range(int(max_node_idx) + 1)]
    #print(adj_list)
    for edge in data.edge_table:
        source = int(edge[2])
        dest = int(edge[3])
        ts = int(edge[1])
        adj_list[source].append(dest)
        adj_time[source].append(ts)
        adj_list[dest].append(source)
        adj_time[dest].append(ts)
    return NeighborFinder(adj_list, adj_time, neighbour_max)


class NeighborFinder:
    def __init__(self, adj_list, adj_time, neighbour_max=None, seed=42):
        self.adj_list = []
        self.adj_time = []
        for i in range(len(adj_list)):
            # We sort the list based on timestamp
            sorted_ids = np.argsort(adj_time[i])
            self.adj_list.append([adj_list[i][ind] for ind in sorted_ids])
            self.adj_time.append([adj_time[i][ind] for ind in sorted_ids])
        self.neighbour_max = neighbour_max
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

    def get_temporal_neighbor_statistics(self, nodes, cur_time):
        to_neighs = [self.adj_list[int(node)] for node in nodes]
        time_neighs = [self.adj_time[int(node)] for node in nodes]

        samp_neighs = []
        dic_temp = set()
        dic_edge = {}
        total_value = {}

        for i in range(len(to_neighs)):
            samp_neighs.append([int(nodes[i])])
            dic_temp.add(int(nodes[i]))
            time_neighs_np = np.array(time_neighs[i])
            max_j = np.searchsorted(time_neighs_np, cur_time)
            if self.neighbour_max is not None and self.neighbour_max < max_j:
                possible_ids = sorted(self.random_state.randint(0, max_j, self.neighbour_max))
            else:
                possible_ids = list(range(max_j))

            samp_neighs[-1].extend([to_neighs[i][j] for j in possible_ids])
            dic_temp.update((to_neighs[i][j] for j in possible_ids))

            new_time = np.exp((time_neighs_np[possible_ids] - cur_time) / 100)
            total_value[i] = float(new_time.sum())
            if total_value[i] == 0:
                total_value[i] = 1
            
            dic_edge[i] = {}
            for j_ind in range(len(possible_ids)):
                if (to_neighs[i][possible_ids[j_ind]] not in dic_edge[i]):
                    dic_edge[i][to_neighs[i][possible_ids[j_ind]]] = new_time[j_ind]
                else:
                    dic_edge[i][to_neighs[i][possible_ids[j_ind]]] += new_time[j_ind]
        return samp_neighs, dic_temp, dic_edge, total_value
