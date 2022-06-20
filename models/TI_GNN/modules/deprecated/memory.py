import torch
from torch import nn


class Memory(nn.Module):

    def __init__(self, n, memory_dimension, device="cpu"):
        super(Memory, self).__init__()
        self.n = n
        self.memory_dimension = memory_dimension
        self.device = device
        self.memory = self.__init_memory__()

    def __init_memory__(self):
        self.memory = nn.Parameter(torch.zeros((self.n, self.memory_dimension)).to(self.device),
                                        requires_grad=False)

    def detach_memory(self):
        self.memory.detach_()

    def backup_memory(self):
        return self.memory.data.clone()

    def restore_memory(self, memory_backup):
        self.memory.data = memory_backup.clone()
