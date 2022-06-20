import numpy as np

from models.TI_GNN.utils.sampling_fn import get_sampling_fn


class DirectedEdgeSampler:
    def __init__(self, sampling_fn_name, n_sample, min_n_sample, max_n_sample):
        self.sampling_fn = get_sampling_fn(sampling_fn_name)
        self.n_sample = n_sample
        self.min_n_sample = min_n_sample
        self.max_n_sample = max_n_sample

    def get_nodes_n_sample(self, in_adj_list: dict):
        return {}

    def sample(self, in_adj_list: dict):
        res_in_adj_list = {}
        node_to_n_sample = self.get_nodes_n_sample(in_adj_list)
        for node in in_adj_list:
            res_in_adj_list[node] = self.sampling_fn(in_adj_list[node], node_to_n_sample[node])
        return res_in_adj_list


class FixedKDirectedEdgeSampler(DirectedEdgeSampler):
    def get_nodes_n_sample(self, in_adj_list: dict):
        res = {}
        for node in in_adj_list:
            res[node] = min(len(in_adj_list[node]), self.n_sample)
        return res


class DegreeDirectedEdgeSampler(DirectedEdgeSampler):
    def get_nodes_n_sample(self, in_adj_list: dict):
        degree_to_n_sample = {x: x for x in range(self.min_n_sample)}
        unique_degrees = np.unique([len(node_adj_list) for node_adj_list in in_adj_list.values() if len(node_adj_list) > 0])
        if len(unique_degrees) > 0:
            mn = max(min(unique_degrees), self.min_n_sample)
            mx = max(unique_degrees)
            if mn != mx:
                n_samples = (unique_degrees - mn)*(self.max_n_sample-self.min_n_sample)/(mx - mn) + self.min_n_sample
                degree_to_n_sample.update(dict(zip(unique_degrees, n_samples)))
            else:
                degree_to_n_sample[mn] = min(mn, self.max_n_sample)
        return {node: degree_to_n_sample[len(node_adj_list)] for node, node_adj_list in in_adj_list.items()}


def get_directed_edge_sampler(sampler_name, sampling_fn_name, n_sample, min_n_sample, max_n_sample):
    if sampler_name == 'fixed':
        return FixedKDirectedEdgeSampler(sampling_fn_name, n_sample, min_n_sample, max_n_sample)
    if sampler_name == 'degree':
        return DegreeDirectedEdgeSampler(sampling_fn_name, n_sample, min_n_sample, max_n_sample)
