import torch.nn as nn
from models.TI_DC_GNN.functions.fusion_fn import get_fusion_fn


class MemoryWriteIn(nn.Module):
    def __init__(self, old_node_memory_dim, updated_memory_dim, fusion_fn_config):
        super().__init__()
        self.old_node_memory_dim = old_node_memory_dim
        self.updated_memory_dim = updated_memory_dim
        self.writer_in = get_fusion_fn(fn_name=fusion_fn_config['fn_name'],
                                     act_fn_name=fusion_fn_config['act_fn_name'],
                                     input_dim=old_node_memory_dim,
                                     hidden_dim=updated_memory_dim)

    def forward(self, old_node_memory, updated_memory):
        return self.writer_in(old_node_memory, updated_memory)

    def get_output_dim(self):
        self.writer_in.get_output_dim(self.old_node_memory_dim, self.updated_memory_dim)


def get_memory_write_in(old_node_memory_dim, updated_memory_dim, fusion_fn_config):
    return MemoryWriteIn(old_node_memory_dim, updated_memory_dim, fusion_fn_config)
