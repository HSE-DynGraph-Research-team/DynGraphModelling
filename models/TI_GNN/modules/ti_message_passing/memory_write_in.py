import torch.nn as nn

from models.TI_GNN.layers.act_fn import get_act_fn


class MemoryWriteIn(nn.Module):
    def __init__(self, old_node_memory_dim, updated_memory_dim, act_fn_name):
        super().__init__()
        self.old_node_memory_dim = old_node_memory_dim
        self.updated_memory_dim = updated_memory_dim
        self.act_fn = get_act_fn(act_fn_name)

    def forward(self, old_node_memory, updated_memory):
        pass

    def get_output_dim(self):
        pass


class IdMemoryWriteIn(MemoryWriteIn):
    def forward(self, old_node_memory, updated_memory):
        return updated_memory

    def get_output_dim(self):
        return self.updated_memory_dim


def get_memory_write_in(memory_write_in_name, old_node_memory_dim, updated_memory_dim, act_fn_name):
    if memory_write_in_name == 'id':
        return IdMemoryWriteIn(old_node_memory_dim, updated_memory_dim, act_fn_name)
