import torch.nn as nn

from models.TI_GNN.layers.act_fn import get_act_fn
from models.TI_GNN.layers.cell_nn import get_cell


class MemoryUpdater(nn.Module):
    def __init__(self, node_memory_dim, message_dim, updater_name, act_fn_name):
        super().__init__()
        self.node_memory_dim = node_memory_dim
        self.message_dim = message_dim
        self.updater = get_cell(updater_name)(message_dim, node_memory_dim)
        self.act_fn = get_act_fn(act_fn_name)

    def forward(self, messages, node_memory):
        # target nodes x message dim
        node_memory = self.updater(messages, node_memory)
        return self.act_fn(node_memory)

    def get_output_dim(self):
        return self.node_memory_dim


def get_memory_updater(node_memory_dim, message_dim, updater_name, act_fn_name):
    return MemoryUpdater(node_memory_dim, message_dim, updater_name, act_fn_name)
