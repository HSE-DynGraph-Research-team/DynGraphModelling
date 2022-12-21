from abc import ABC, abstractmethod

import torch.nn as nn

class Function(nn.Module, ABC):
    @abstractmethod
    def __init__(self, *args):
        super().__init__()

    @abstractmethod
    def get_output_dim(self, *args):
        pass

    @abstractmethod
    def forward(self, *args):
        pass