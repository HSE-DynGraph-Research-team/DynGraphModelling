import torch.nn as nn

from models.TI_GNN.layers.act_fn import get_act_fn
from models.TI_GNN.layers.cell_nn import get_cell
from models.TI_GNN.layers.fusion_fn import get_fusion_fn


class BaseMessageBuilder(nn.Module):
    def __init__(self, node_features_dim, edge_features_dim,
                 time_encoding_dim=None, message_cell_name="RNN", time_fusion_fn_name=None, time_input=True, act_fn_name='id'):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim
        self.time_encoding_dim = time_encoding_dim
        self.message_cell_fn = get_cell(message_cell_name)
        self.use_time_encoding = time_encoding_dim > 0
        self.time_fusion_fn = get_fusion_fn(time_fusion_fn_name)
        self.time_input = time_input
        self.act_fn = get_act_fn(act_fn_name)

        # message builder
        input_dim, hidden_dim = self.get_input_hidden_dim(node_features_dim, edge_features_dim)
        if self.use_time_encoding:
            if time_input:
                input_dim = self.time_fusion_fn.get_output_dim(input_dim, time_encoding_dim)
            else:
                hidden_dim = self.time_fusion_fn.get_output_dim(hidden_dim, time_encoding_dim)
        self.message_builder = self.message_cell_fn(input_dim, hidden_dim)

    def forward(self, node_features, edge_features, time_encoding):
        # node_features: nxf
        input, hidden = self.get_input_hidden_order(node_features, edge_features)
        if self.use_time_encoding:
            if self.time_input:
                input = self.time_fusion_fn(input, time_encoding)
            else:
                hidden = self.time_fusion_fn(hidden, time_encoding)
        return self.message_builder(input, hidden)

    def get_input_hidden_order(self, node_attrs, edge_attrs):
        return 0, 0

    def get_output_dim(self):
        return self.message_cell_fn.get_output_dim()


class NodeInputMessageBuilder(BaseMessageBuilder):
    def get_input_hidden_order(self, node_attrs, edge_attrs):
        return node_attrs, edge_attrs


class NodeHiddenMessageBuilder(BaseMessageBuilder):
    def get_input_hidden_order(self, node_attrs, edge_attrs):
        return edge_attrs, node_attrs


def get_message_builder(node_input, node_features_dim, edge_features_dim, time_encoding_dim, message_cell_name, time_fusion_fn_name, time_input, act_fn_name):
    if node_input:
        return NodeInputMessageBuilder(node_features_dim, edge_features_dim, time_encoding_dim, message_cell_name, time_fusion_fn_name, time_input, act_fn_name)
    return NodeHiddenMessageBuilder(node_features_dim, edge_features_dim, time_encoding_dim, message_cell_name, time_fusion_fn_name, time_input, act_fn_name)

