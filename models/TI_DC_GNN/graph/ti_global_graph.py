import dgl
import numpy as np

from models.TI_GNN_old.utils.graph_utils import build_conseq_in_adj_list_from_causal


class TiGlobalGraph:
    def __init__(self, dual_graph_causal, dual_graph_conseq):
        self.dual_graph_causal = dual_graph_causal
        self.dual_graph_conseq = dual_graph_conseq


class TiGlobalGraphBuilder:

    def __init__(self, edge_sampler):
        self.edge_sampler = edge_sampler

    def build(self, ti_dual_adj_list):
        causal_in_adj_list = ti_dual_adj_list.causal_in_adj_list
        causal_in_adj_list = self.edge_sampler.sample(causal_in_adj_list)
        conseq_in_adj_list = build_conseq_in_adj_list_from_causal(causal_in_adj_list)
        return TiGlobalGraph(self.in_adj_list_to_dgl(causal_in_adj_list), self.in_adj_list_to_dgl(conseq_in_adj_list))


    def in_adj_list_to_dgl(self, in_adj_list):
        sources, destinations, edge_ids = [], [], []
        for dest in in_adj_list:
            for source, edge_id in in_adj_list[dest]:
                sources.append(source)
                edge_ids.append(edge_id)
            destinations.extend([dest for _ in range(len(in_adj_list[dest]))])
        sorted_ids = np.argsort(edge_ids)
        dual_graph = dgl.graph(np.array(sources)[sorted_ids], np.array(destinations)[sorted_ids])
        return dual_graph
