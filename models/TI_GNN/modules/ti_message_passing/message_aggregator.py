import torch
import torch.nn as nn

from models.TI_GNN.layers.act_fn import get_act_fn
from models.TI_GNN.layers.aggregation_nn import get_aggregator


class MessageAggregator(nn.Module):
    def __init__(self, node_features_dim, message_dim, aggregation_fn_name, act_fn_name):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.message_dim = message_dim
        self.aggregation_fn = get_aggregator(aggregation_fn_name)(message_dim, node_features_dim)
        self.act_fn = get_act_fn(act_fn_name)

    def forward(self, node_features, grouped_messages: list):
        # messages num x message dim
        res = []
        for i in range(len(grouped_messages)):
            res.append(self.aggregation_fn(grouped_messages[i], node_features))
        return self.act_fn(torch.cat(res, dim=0))

    def get_output_dim(self):
        return self.aggregation_fn.get_output_dim()


def get_message_aggregator(node_features_dim, message_dim, aggregation_fn_name, act_fn_name, agg_name):
    return MessageAggregator(node_features_dim, message_dim, aggregation_fn_name, act_fn_name)
