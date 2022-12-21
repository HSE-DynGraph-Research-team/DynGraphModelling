import torch.nn as nn

from models.TI_DC_GNN.functions.single_transform_fn import get_single_transform_fn
from models.TI_DC_GNN.functions.fusion_fn import get_fusion_fn


class MemoryReadOut(nn.Module):
    def __init__(self, memory_dim, features_dim, fusion_fn_config, transform_memory_fn_config, transform_feats_fn_config):
        super().__init__()
        self.memory_dim = memory_dim
        self.features_dim = features_dim
        self.transform_memory_fn = get_single_transform_fn(fn_name=transform_memory_fn_config['fn_name'],
                                                           act_fn_name=transform_memory_fn_config['act_fn_name'],
                                                           input_dim=memory_dim,
                                                           output_dim=transform_memory_fn_config['output_dim'])
        self.transform_feats_fn = get_single_transform_fn(fn_name=transform_feats_fn_config['fn_name'],
                                                           act_fn_name=transform_feats_fn_config['act_fn_name'],
                                                           input_dim=features_dim,
                                                           output_dim=transform_feats_fn_config['output_dim'])
        self.fusion_fn = get_fusion_fn(fn_name=fusion_fn_config['fn_name'],
                                       act_fn_name=fusion_fn_config['act_fn_name'],
                                       input_dim=self.transform_feats_fn.get_output_dim(),
                                       hidden_dim=self.transform_memory_fn.get_output_dim())

    def forward(self, memory, features):
        transformed_memory = self.transform_memory_fn(memory)
        transformed_features = self.transform_feats_fn(features)
        return self.fusion_fn(transformed_features, transformed_memory)

    def get_output_dim(self):
        transformed_memory = self.transform_memory_fn.get_output_dim()
        transformed_features = self.transform_feats_fn.get_output_dim()
        return self.fusion_fn.get_output_dim(transformed_memory, transformed_features)


def get_memory_readout(memory_dim, features_dim, fusion_fn_config, transform_memory_fn_config, transform_feats_fn_config):
    return MemoryReadOut(memory_dim, features_dim, fusion_fn_config,
                         transform_memory_fn_config, transform_feats_fn_config)
