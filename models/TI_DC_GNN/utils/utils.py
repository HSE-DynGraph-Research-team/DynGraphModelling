from dataclasses import dataclass
from typing import Union

import numpy as np
import functools
import operator

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


def flatten_list(a):
    return functools.reduce(operator.iconcat, a, [])


def get_all_edges(edge_table):
    sources, destinations = edge_table[:, 2], edge_table[:, 3]
    return sources, destinations


@dataclass
class DimensionConfig:
    node_memory_dim: int
    node_features_dim: int
    edge_memory_dim: int
    edge_features_dim: int


@dataclass
class FeatureProperties:
    node_features: Union[torch.Tensor, np.ndarray]
    edge_features: Union[torch.Tensor, np.ndarray]


@dataclass
class MemoryProperties:
    node_memory: Union[torch.Tensor, np.ndarray]
    edge_memory: Union[torch.Tensor, np.ndarray]