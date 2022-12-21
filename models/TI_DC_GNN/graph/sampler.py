import random
from abc import ABC, abstractmethod

import numpy as np

from models.TI_DC_GNN.utils.sampling_fn import get_sampling_fn


class DirectedNodeSampler(ABC):
    def __init__(self, sampling_fn_name, min_n_sample, max_n_sample):
        self.sampling_fn = get_sampling_fn(sampling_fn_name)
        self.min_n_sample = min_n_sample
        self.max_n_sample = max_n_sample
        self.round_fns = [np.floor, np.ceil]

    @abstractmethod
    def get_n_samples(self, sum_degrees, *args):
        pass

    @abstractmethod
    def get_node_to_n_sample_args(self, node_to_grouped_predecessors):
        pass

    def get_nodes_n_sample_by_edge(self, node_to_grouped_predecessors: dict):
        res = {}
        node_to_n_sample_args = self.get_node_to_n_sample_args(node_to_grouped_predecessors)
        for node in node_to_grouped_predecessors:
            if len(node_to_grouped_predecessors[node]) == 0:
                continue
            res[node] = {}
            degrees = [len(nodes) for nodes in node_to_grouped_predecessors[node].values()]
            round_fns_ids = np.random.permutation(2)
            round_fns = [self.round_fns[ind] for ind in round_fns_ids]
            sum_degrees = sum(degrees)
            n_sample = self.get_n_samples(sum_degrees, *node_to_n_sample_args[node])
            for i, edge in enumerate(node_to_grouped_predecessors[node]):
                if n_sample < sum_degrees:
                    res[node][edge] = round_fns[i](degrees[i] / sum_degrees * n_sample)
                else:
                    res[node][edge] = degrees[i]
        return res

    def sample(self, node_to_grouped_predecessors: dict):
        res_node_to_grouped_predecessors = {}
        node_to_n_sample_by_edge = self.get_nodes_n_sample_by_edge(node_to_grouped_predecessors)
        for node in node_to_grouped_predecessors:
            res_node_to_grouped_predecessors[node] = {}
            for edge in node_to_grouped_predecessors:
                res_node_to_grouped_predecessors[node][edge] =\
                    self.sampling_fn(res_node_to_grouped_predecessors[node][edge], node_to_n_sample_by_edge[node][edge])
        return res_node_to_grouped_predecessors


class FixedKDirectedEdgeSampler(DirectedNodeSampler):
    def get_n_samples(self, sum_degrees, *args):
        return random.randint(self.min_n_sample, self.max_n_sample)

    def get_node_to_n_sample_args(self, node_to_grouped_predecessors):
        return []


class DegreeDirectedEdgeSampler(DirectedNodeSampler):
    def get_n_samples(self, sum_degrees, degree_to_n_sample):
        return degree_to_n_sample[sum_degrees]

    def get_node_to_n_sample_args(self, node_to_grouped_predecessors):
        degree_to_n_sample = {x: x for x in range(self.min_n_sample)}
        degrees = [len(node_adj_list) for node_adj_list in node_to_grouped_predecessors.values() if
                   len(node_adj_list) > 0]
        unique_degrees = np.unique(degrees)
        if len(unique_degrees) > 0:
            mn = max(min(unique_degrees), self.min_n_sample)
            mx = max(unique_degrees)
            if mn != mx:
                n_samples = (unique_degrees - mn) * (self.max_n_sample - self.min_n_sample) / (
                            mx - mn) + self.min_n_sample
                degree_to_n_sample.update(dict(zip(unique_degrees, n_samples)))
            else:
                degree_to_n_sample[mn] = min(mn, self.max_n_sample)
        return [degree_to_n_sample]


def get_directed_node_sampler(sampler_name, sampling_fn_name, min_n_sample, max_n_sample):
    if sampler_name == 'fixed':
        return FixedKDirectedEdgeSampler(sampling_fn_name, min_n_sample, max_n_sample)
    if sampler_name == 'degree':
        return DegreeDirectedEdgeSampler(sampling_fn_name, min_n_sample, max_n_sample)


def get_predecessor_samplers(sampler_config):
    samplers = []
    for sampling_fn_name in sampler_config['sampling_fn_names']:
        for sampler_name in sampler_config['sampler_names']:
            samplers.append(get_directed_node_sampler(sampler_name,
                                                      sampling_fn_name,
                                                      min_n_sample=sampler_config['min_n_sample'],
                                                      max_n_sample=sampler_config['max_n_sample']))
    return samplers
