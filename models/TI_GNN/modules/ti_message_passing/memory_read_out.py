import torch.nn as nn

from models.TI_GNN.layers.act_fn import get_act_fn
from models.TI_GNN.layers.cell_nn import get_cell
from models.TI_GNN.layers.fusion_fn import get_fusion_fn


class MemoryReadOut(nn.Module):
    def __init__(self, node_memory_dim, node_features_dim, act_fn_name):
        super().__init__()
        self.node_memory_dim = node_memory_dim
        self.node_features_dim = node_features_dim
        self.act_fn = get_act_fn(act_fn_name)

    def forward(self, node_memory, node_features):
        pass

    def get_output_dim(self):
        pass


class IdMemoryReadOut(MemoryReadOut):
    def forward(self, node_memory, node_features):
        return self.act_fn(node_memory)


class FusionMemoryReadOut(MemoryReadOut):
    def __init__(self, node_memory_dim, node_features_dim, act_fn_name, fusion_fn_name):
        super().__init__(node_memory_dim, node_features_dim, act_fn_name)
        self.fusion_fn = get_fusion_fn(fusion_fn_name)

    def forward(self, node_memory, node_features):
        return self.act_fn(self.fusion_fn(node_memory, node_features))

    def get_output_dim(self):
        return self.fusion_fn.get_output_dim(self.node_memory_dim, self.node_features_dim)


class TransformMemoryReadOut(MemoryReadOut):
    def __init__(self, node_memory_dim, node_features_dim, act_fn_name, transform_fn_name):
        super().__init__(node_memory_dim, node_features_dim, act_fn_name)
        self.transform_fn = get_cell(transform_fn_name)

    def forward(self, node_memory, node_features):
        node_memory = self.transform_fn(node_memory)
        return self.act_fn(self.transform_fn(node_memory, node_features))

    def get_output_dim(self):
        return self.transform_fn.get_output_dim(self.node_memory_dim, self.node_features_dim)


class TransformFusionMemoryReadOut(MemoryReadOut):
    def __init__(self, node_memory_dim, node_features_dim, act_fn_name, fusion_fn_name, transform_fn_name):
        super().__init__(node_memory_dim, node_features_dim, act_fn_name)
        self.fusion_fn = get_fusion_fn(fusion_fn_name)
        self.transform_fn = get_cell(transform_fn_name)

    def forward(self, node_memory, node_features):
        node_memory = self.act_fn(self.transform_fn(node_memory))
        return self.fusion_fn(node_memory, node_features)

    def get_output_dim(self):
        return self.fusion_fn.get_output_dim(self.node_memory_dim, self.node_features_dim)


def get_memory_readout(memory_readout_name, node_memory_dim, node_features_dim, act_fn_name, fusion_fn_name, transform_fn_name):
    if memory_readout_name == 'id':
        return IdMemoryReadOut(node_memory_dim, node_features_dim, act_fn_name)
    if memory_readout_name == 'fusion':
        return FusionMemoryReadOut(node_memory_dim, node_features_dim, act_fn_name, fusion_fn_name)
    if memory_readout_name == 'transform':
        return TransformMemoryReadOut(node_memory_dim, node_features_dim, act_fn_name, transform_fn_name)
    if memory_readout_name == 'transform_fusion':
        return TransformFusionMemoryReadOut(node_memory_dim, node_features_dim, act_fn_name, fusion_fn_name, transform_fn_name)