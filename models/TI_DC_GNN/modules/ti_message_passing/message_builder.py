from abc import abstractmethod, ABC

import torch.nn as nn
from models.TI_DC_GNN.functions.fusion_fn import get_fusion_fn


class BaseMessageBuilder(nn.Module, ABC):
    def __init__(self, node_features_dim, edge_features_dim, time_encoding_dim,
                 node_time_fusion_fn_config, edge_fusion_fn_config):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim
        self.time_encoding_dim = time_encoding_dim
        self.node_time_fusion_fn = get_fusion_fn(fn_name=node_time_fusion_fn_config['fn_name'],
                                                   act_fn_name=node_time_fusion_fn_config['act_fn_name'],
                                                   input_dim=time_encoding_dim,
                                                   hidden_dim=node_features_dim)
        node_time_dim = self.node_time_fusion_fn.get_output_dim(time_encoding_dim, node_features_dim)
        input_dim, hidden_dim = self.get_input_hidden_order(node_time_dim, edge_features_dim)
        self.edge_fusion_fn = get_fusion_fn(fn_name=edge_fusion_fn_config['fn_name'],
                                             act_fn_name=edge_fusion_fn_config['act_fn_name'],
                                             input_dim=input_dim,
                                             hidden_dim=hidden_dim)

    def forward(self, node_features, edge_features, time_encoding):
        # node_features: nxf
        node_time_features = self.node_time_fusion_fn(node_features, time_encoding)
        input, hidden = self.get_input_hidden_order(node_time_features, edge_features)
        return self.edge_fusion_fn(input, hidden)

    @abstractmethod
    def get_input_hidden_order(self, node_attrs, edge_attrs):
        pass

    def get_output_dim(self):
        return self.edge_fusion_fn.get_output_dim()


class NodeInputMessageBuilder(BaseMessageBuilder):
    def get_input_hidden_order(self, node_attrs, edge_attrs):
        return node_attrs, edge_attrs


class NodeHiddenMessageBuilder(BaseMessageBuilder):
    def get_input_hidden_order(self, node_attrs, edge_attrs):
        return edge_attrs, node_attrs


def get_message_builder(node_input, node_features_dim, edge_features_dim, time_encoding_dim, node_time_fusion_fn_config,
                        edge_fusion_fn_config):
    if node_input:
        return NodeInputMessageBuilder(node_features_dim=node_features_dim,
                                       edge_features_dim=edge_features_dim,
                                       time_encoding_dim=time_encoding_dim,
                                       node_time_fusion_fn_config=node_time_fusion_fn_config,
                                       edge_fusion_fn_config=edge_fusion_fn_config)
    return NodeHiddenMessageBuilder(node_features_dim=node_features_dim,
                                    edge_features_dim=edge_features_dim,
                                    time_encoding_dim=time_encoding_dim,
                                    node_time_fusion_fn_config=node_time_fusion_fn_config,
                                    edge_fusion_fn_config=edge_fusion_fn_config)

