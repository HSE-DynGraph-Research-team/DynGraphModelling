import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TI_DC_GNN.functions.transform_fn import TransformFn


class AggregationFn(TransformFn):
    def __init__(self, act_fn_name, target_node_features_dim, message_dim):
        super().__init__(act_fn_name)
        self.target_node_features_dim = target_node_features_dim
        self.message_dim = message_dim

    def _forward(self, target_node_features, grouped_messages, grouped_message_timestamps):
        # target_node_features: batch_size x feat_size
        # grouped_messages: batch_size x messages_num x message_dim
        # grouped_message_timestamps: batch_size x messages_num
        pass

    def get_output_dim(self):
        pass


class MeanAggregationFn(AggregationFn):
    def _forward(self, target_node_features, grouped_messages, grouped_message_timestamps):
        return torch.mean(grouped_messages, dim=1)

    def get_output_dim(self):
        return self.message_dim


class MeanWithWeightsAggregationFn(AggregationFn):
    def __init__(self, act_fn_name, target_node_features_dim, message_dim, output_dim):
        super().__init__(act_fn_name, target_node_features_dim, message_dim)
        self.output_dim = output_dim
        self.fc = nn.Linear(message_dim, output_dim)

    def _forward(self, target_node_features, grouped_messages, grouped_message_timestamps):
        mean_messages = torch.mean(grouped_messages, dim=1)
        return self.fc(mean_messages)

    def get_output_dim(self):
        return self.output_dim


class ExpTimeAggregationFn(AggregationFn):
    def _forward(self, target_node_features, grouped_messages, grouped_message_timestamps):
        time_weights = torch.softmax(grouped_message_timestamps, dim=1)
        agg_messages = torch.matmul(grouped_messages, time_weights).sum(dim=1)  # batch x message_dim
        return agg_messages

    def get_output_dim(self):
        return self.message_dim


class ExpTimeWithWeightsAggregationFn(AggregationFn):
    def __init__(self, act_fn_name, target_node_features_dim, message_dim, output_dim):
        super().__init__(act_fn_name, target_node_features_dim, message_dim)
        self.output_dim = output_dim
        self.fc = nn.Linear(message_dim, output_dim)

    def forward(self, target_node_features, grouped_messages, grouped_message_timestamps):
        time_weights = F.softmax(grouped_message_timestamps, dim=1)
        agg_messages = (grouped_messages * time_weights).sum(dim=1)  # batch x message_dim
        return self.fc(agg_messages)

    def get_output_dim(self):
        return self.output_dim


class AttentionAggregarionFn(AggregationFn):
    def __init__(self, act_fn_name, target_node_features_dim, message_dim):
        super().__init__(act_fn_name, target_node_features_dim, message_dim)
        self.attn_fc = nn.Linear(target_node_features_dim + message_dim, 1, bias=False)

    def _forward(self, target_node_features, grouped_messages, grouped_message_timestamps):
        attn_weghts = self.attn_fc(torch.cat([target_node_features, grouped_messages], dim=0)) # batch x message_num
        return (grouped_messages * attn_weghts).sum(dim=1)  # batch x message_dim

    def get_output_dim(self):
        return self.message_dim


class AttentionWithWeightsAggregarionFn(AggregationFn):
    def __init__(self, act_fn_name, target_node_features_dim, message_dim, output_dim):
        super().__init__(act_fn_name, target_node_features_dim, message_dim)
        self.fc = nn.Linear(message_dim, output_dim, bias=False)
        self.attn_fc = nn.Linear(target_node_features_dim + output_dim, 1, bias=False)
        self.output_dim = output_dim

    def _forward(self, target_node_features, grouped_messages, grouped_message_timestamps):
        grouped_messages = self.fc(grouped_messages)
        attn_weghts = self.attn_fc(torch.cat([target_node_features, grouped_messages], dim=0)) # batch x message_num
        return (grouped_messages * attn_weghts).sum(dim=1)  # batch x message_dim

    def get_output_dim(self):
        return self.output_dim


def get_aggregator(fn_name, act_fn_name, target_node_features_dim, message_dim, output_dim):
    if fn_name is None:
        return None
    if fn_name == 'mean':
        return MeanAggregationFn(act_fn_name, target_node_features_dim, message_dim)
    if fn_name == 'mean_weights':
        return MeanWithWeightsAggregationFn(act_fn_name, target_node_features_dim, message_dim, output_dim)
    if fn_name == 'exp':
        return ExpTimeAggregationFn(act_fn_name, target_node_features_dim, message_dim)
    if fn_name == 'exp_weigths':
        return ExpTimeWithWeightsAggregationFn(act_fn_name, target_node_features_dim, message_dim, output_dim)
    if fn_name == 'attn':
        return AttentionAggregarionFn(act_fn_name, target_node_features_dim, message_dim)
    if fn_name == 'attn_weights':
        return AttentionWithWeightsAggregarionFn(act_fn_name, target_node_features_dim, message_dim, output_dim)