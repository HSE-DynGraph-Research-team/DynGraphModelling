import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import random
import numpy as np
import gc

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, ngh_finder, features, device, gcn=False, time=0):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        time --- the current time
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.ngh_finder = ngh_finder
        self.device = device
        self.gcn = gcn
        self.time = time

    def forward(self, nodes):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        time_neighs --- number of neighbors to sample with its appearance's time. No sampling if None.
        """
        if type(nodes) == torch.Tensor:
            nodes = nodes.cpu().numpy()
        samp_neighs, dic_temp, dic_edge, total_value =\
            self.ngh_finder.get_temporal_neighbor_statistics(nodes, self.time)

        """samp_neighs = []
        dic_temp = set()
        dic_edge = {}
        total_value = {}
        #print(nodes)
        if type(nodes)==torch.Tensor:
            nodes=nodes.cpu()
        nodes = np.array(nodes)
        for i in range(0, len(to_neighs)):
            samp_neighs.append([int(nodes[i])])
            dic_edge[i] = {}
            dic_temp.add(int(nodes[i]))

            max_j = np.searchsorted(time_neighs[i], self.time, side='right')
            new_time = np.exp((time_neighs[i][:max_j] - self.time) / 100)
            total_value[i] = float(new_time.sum())
            if total_value[i] == 0:
                total_value[i] = 1

            samp_neighs[-1].extend([to_neighs[i][j] for j in range(max_j)])
            dic_temp.update((to_neighs[i][j] for j in range(max_j)))

            for j in range(max_j):
                if (to_neighs[i][j] not in dic_edge[i]):
                    dic_edge[i][to_neighs[i][j]] = new_time[j]
                else:
                    dic_edge[i][to_neighs[i][j]] += new_time[j]"""

        unique_nodes_list = list(dic_temp) #list([key for key in dic_temp])
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes))).to(self.device)

        # column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # mask[row_indices, column_indices] = 1
        for key1 in dic_edge:
            for key2 in dic_edge[key1]:
                mask[key1, unique_nodes[key2]] = dic_edge[key1][key2] / total_value[key1]
        for i in range(0, len(nodes)):
            # print(mask[i][unique_nodes[nodes[i]]])
            mask[i, unique_nodes[nodes[i]]] += 1

        #mask = mask.to(self.device)
        # print("Type-----------------------------------------:",mask.type)
        num_neigh = mask.sum(1, keepdim=True)
        # num_neigh = mask.sum(1)
        # print("num_neigh:"+str(num_neigh))
        mask = mask.div(num_neigh)
        embed_matrix = self.features(torch.LongTensor(unique_nodes_list).to(self.device))
        # print("embed_matrix:"+str(embed_matrix))
        to_feats = mask.mm(embed_matrix)
        #del mask, embed_matrix, dic_edge, dic_temp, total_value
        # gc.collect()
        return to_feats

    def set_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder