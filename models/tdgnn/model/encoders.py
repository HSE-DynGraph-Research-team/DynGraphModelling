import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import gc
class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim,
            embed_dim, aggregator, device,
            num_sample=10,
            base_model=None, gcn=False,
            feature_transform=False,last=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.last=last
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.device = device
        self.aggregator.device = device
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        #print(nodes)
        neigh_feats = self.aggregator.forward(nodes)

        #print(neigh_feats)
        if not self.gcn:
            self_feats = self.features(torch.LongTensor(nodes).to(self.device))
            combined = torch.cat([self_feats, neigh_feats], dim=1)

        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        del neigh_feats
        # gc.collect()
        return combined
