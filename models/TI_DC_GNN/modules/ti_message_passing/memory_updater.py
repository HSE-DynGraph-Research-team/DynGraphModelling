import torch.nn as nn

from models.TI_DC_GNN.functions.fusion_fn import get_fusion_fn


class MemoryUpdater(nn.Module):
    def __init__(self, memory_dim, message_dim, fusion_fn_config):
        super().__init__()
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.updater = get_fusion_fn(fn_name=fusion_fn_config['fn_name'],
                                     act_fn_name=fusion_fn_config['act_fn_name'],
                                     input_dim=message_dim,
                                     hidden_dim=memory_dim)

    def forward(self, messages, node_memory):
        # target nodes x message dim
        return self.updater(messages, node_memory)

    def get_output_dim(self):
        return self.updater.get_output_dim(self.message_dim, self.memory_dim)


def get_memory_updater(memory_dim, message_dim, fusion_fn_config):
    return MemoryUpdater(memory_dim, message_dim, fusion_fn_config)
