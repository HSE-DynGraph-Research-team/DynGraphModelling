import torch
import torch.nn as nn

from models.TI_DC_GNN.functions.aggregation_fn import get_aggregator
from models.TI_DC_GNN.functions.fusion_fn import get_fusion_fn


class MessageAggregator(nn.Module):
    def __init__(self, target_node_features_dim, message_dim, target_time_fusion_config, aggregation_config):
        super().__init__()
        self.target_node_features_dim = target_node_features_dim
        self.message_dim = message_dim
        self.target_time_fusion_fn = get_fusion_fn(fn_name=target_time_fusion_config['fn_name'],
                                                   act_fn_name=target_time_fusion_config['act_fn_name'],
                                                   input_dim=1,
                                                   hidden_dim=1)
        self.aggregation_fn = get_aggregator(fn_name=aggregation_config['fn_name'],
                                             act_fn_name=aggregation_config['act_fn_name'],
                                             target_node_features_dim=target_node_features_dim,
                                             message_dim=message_dim,
                                             output_dim=aggregation_config['output_dim'])

    def forward(self, target_node_features, node_timestamps, grouped_messages, grouped_message_timestamps):
        # node_features: batch_size x feat_size
        # node_timestamps: batch_size
        # grouped_messages: batch_size x messages_num x message_dim
        # grouped_message_timestamps: batch_size x messages_num

        grouped_node_timestamps = node_timestamps.unsqueeze(1)
        grouped_message_timestamps = self.target_time_fusion_fn(grouped_node_timestamps, grouped_message_timestamps)
        return self.aggregation_fn(target_node_features, grouped_messages, grouped_message_timestamps)

        # res = []
        # for i in range(len(grouped_messages)):
        #     res.append(self.aggregation_fn(grouped_messages[i], node_features))
        # return self.act_fn(torch.cat(res, dim=0))

    def get_output_dim(self):
        return self.aggregation_fn.get_output_dim()


def get_message_aggregator(target_node_features_dim, message_dim, target_time_fusion_config, aggregation_config):
    return MessageAggregator(target_node_features_dim, message_dim, target_time_fusion_config, aggregation_config)
