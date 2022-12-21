from typing import Optional, Tuple
import torch.nn as nn

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor

from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class ConvGCNLayer(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cached: bool = False,
                 normalize: bool = True,
                 add_self_loops: bool = False,
                 improved: bool = False,
    ):

        super(ConvGCNLayer, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self.reset_parameters()

    def reset_parameters(self):
       pass

    def normalize_adj(self, num_nodes: int, edge_index: Adj, edge_weight: OptTensor):
        if isinstance(edge_index, Tensor):
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, num_nodes,
                    self.improved, self.add_self_loops)
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]

        elif isinstance(edge_index, SparseTensor):
            cache = self._cached_adj_t
            if cache is None:
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, num_nodes,
                    self.improved, self.add_self_loops)
                if self.cached:
                    self._cached_adj_t = edge_index
            else:
                edge_index = cache
        return edge_index, edge_weight

    def forward(self, x: Tensor, weight, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            edge_index, edge_weight = self.normalize_adj(x.size(self.node_dim), edge_index, edge_weight)

        x = x @ weight

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class SimpleEmbeddingMemory(nn.Module):

    def __init__(self):
        super().__init__()
        self.h = nn.L

    def forward(self, node_index):
        pass