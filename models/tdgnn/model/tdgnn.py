import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
import torch.nn.functional as F
from .encoders import Encoder
from .aggregators import MeanAggregator


class TDGNN_GraphSage(nn.Module):

    def __init__(self,feat_data, ngh_finder, device,
                 layer_num=2, embed_dim=128, edge_agg_name='mean', num_classes=2, num_sample=None):
        super(TDGNN_GraphSage, self).__init__()
        print(feat_data)
        self.features = nn.Embedding(feat_data.shape[0], feat_data.shape[1]).to(device)
        #self.features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
        self.features.weight = nn.Parameter(torch.randn((feat_data.shape[0], feat_data.shape[1])), requires_grad=False)
        #self.ngh_finder = ngh_finder
        # features.cuda()
        self.aggs = [MeanAggregator(ngh_finder, self.features, device=device)]
        self.encs = [Encoder(self.features, feat_data.shape[1], embed_dim, self.aggs[-1],
                           device=device, num_sample=num_sample, gcn=True)]

        for i in range(layer_num-1):
            prev_enc = self.encs[-1]
            self.aggs.append(MeanAggregator(ngh_finder, lambda nodes: prev_enc(nodes).t(), device=device))
            self.encs.append(Encoder(lambda nodes: prev_enc(nodes).t(),  prev_enc.embed_dim, embed_dim, self.aggs[-1], num_sample=num_sample, base_model=prev_enc, gcn=True, device=device))
        self.enc = self.encs[-1]

        """self.agg1 = MeanAggregator(ngh_finder, self.features, device=device)
        self.enc1 = Encoder(self.features, feat_data.shape[1], embed_dim, self.agg1,
                           device=device, num_sample=num_sample, gcn=True)
        self.agg2 = MeanAggregator(ngh_finder, lambda nodes: self.enc1(nodes).t(), device=device)
        self.enc = Encoder(lambda nodes: self.enc1(nodes).t(),  self.enc1.embed_dim, embed_dim, self.agg2, num_sample=num_sample, base_model=self.enc1, gcn=True, device=device)
        self.aggs = [self.agg1, self.agg2]"""

        """self.agg1 = MeanAggregator(ngh_finder, self.features, device=device)
        self.enc = Encoder(self.features, feat_data.shape[1], embed_dim, self.agg1,
                           device=device, num_sample=num_sample, gcn=True)
        self.aggs = [self.agg1]"""

        self.enc.last = True
        self.edge_agg_name = edge_agg_name
        self.num_classes = num_classes
        self.device = device
        self.init_graphsage_params()

    def init_graphsage_params(self):
        self.xent = nn.CrossEntropyLoss()
        if self.edge_agg_name != "activation" and self.edge_agg_name != "origin":
            self.weight = nn.Parameter(torch.FloatTensor(self.num_classes, self.enc.embed_dim))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(self.num_classes, self.enc.embed_dim * 2))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """nodes: [edge[0], edge[1]]"""
        #print(nodes)
        embeds = self.enc(nodes)
        embeds = self.pool_embeds(embeds)
        return self.weight.mm(embeds).t()

    def pool_embeds(self, embeds):
        #mean
        if self.edge_agg_name=="mean":
            embeds=(embeds[:,0]+embeds[:,1])/2
            embeds = embeds.unsqueeze(1)
        #hadamard
        elif self.edge_agg_name=='had':
            embeds = embeds[:,0].mul(embeds[:,1])
            embeds = embeds.unsqueeze(1)
        #weight-l1
        elif self.edge_agg_name=="w1":
            embeds = torch.abs(embeds[:,0]-embeds[:,1])
            embeds = embeds.unsqueeze(1)
        #weight-l2
        elif self.edge_agg_name=="w2":
            embeds=torch.abs(embeds[:,0]-embeds[:,1]).mul(torch.abs(embeds[:,0]-embeds[:,1]))
            embeds = embeds.unsqueeze(1)
        #activation
        elif self.edge_agg_name=='activation':
            embeds = torch.cat((embeds[:, 0], embeds[:, 1]), 0).unsqueeze(1)
            embeds = F.relu(embeds)
        # original
        elif self.edge_agg_name=='origin':
            embeds = torch.cat((embeds[:, 0], embeds[:, 1]), 0).unsqueeze(1)
        return embeds

    def loss_predict(self, edges, labels=None):
        for j in range(len(self.aggs)):
            self.aggs[j].time = edges[0][2]
        scores = self.forward([edges[0][0], edges[0][1]])
        predict_y = [scores.cpu().detach().numpy()[0][1]]
        for i in range(1, len(edges)):
            for j in range(len(self.aggs)):
                self.aggs[j].time = edges[i][2]
            temp = self.forward([edges[i][0], edges[i][1]])
            #temp = F.softmax(temp, dim=1) Don't understand for what
            predict_y.append(temp.cpu().detach().numpy()[0][1])
            scores = torch.cat((scores, temp), 0)
        if labels is None:
            return None, predict_y
        return self.xent(scores, labels), predict_y

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, timestamps):
        edges = np.array([source_nodes, destination_nodes]).T
        for j in range(len(self.aggs)):
            self.aggs[j].time = timestamps[0]
        temp = self.enc(edges[0])
        source_embeds = temp[:,0].unsqueeze(0)
        destination_embeds = temp[:,1].unsqueeze(0)
        for i in range(1, len(edges)):
            for j in range(len(self.aggs)):
                self.aggs[j].time = timestamps[i]
            temp = self.enc(edges[i])
            source_embeds = torch.cat((source_embeds, temp[:,0].unsqueeze(0)), 0)
            destination_embeds = torch.cat((destination_embeds, temp[:,1].unsqueeze(0)), 0)

        return source_embeds, destination_embeds

    def compute_edge_probabilities(self, source_nodes, destination_nodes,
                                   negative_nodes, timestamps, pos_neg_labels=None):
        #print(pos_neg_labels)
        #print(type(source_nodes))
        pos_edges = np.array([source_nodes, destination_nodes]).T
        neg_edges = np.array([source_nodes, negative_nodes]).T
        #pos_probs = []
        #neg_probs = []

        for j in range(len(self.aggs)):
            self.aggs[j].time = timestamps[0]
        pos_scores = self.forward([pos_edges[0][0], pos_edges[0][1]])
        neg_scores = self.forward([neg_edges[0][0], neg_edges[0][1]])

        #pos_probs.append(pos_scores[-1].cpu().detach().numpy()[0][1])
        #neg_probs.append(neg_scores.cpu().detach().numpy()[0][1])

        for i in range(1, len(pos_edges)):
            for j in range(len(self.aggs)):
                self.aggs[j].time = timestamps[i]
            pos_temp = self.forward([pos_edges[i][0], pos_edges[i][1]])
            neg_temp = self.forward([neg_edges[i][0], neg_edges[i][1]])

            #pos_temp = F.softmax(pos_temp, dim=1)
            #neg_temp = F.softmax(neg_temp, dim=1)

            pos_scores = torch.cat((pos_scores, pos_temp), 0)
            neg_scores = torch.cat((neg_scores, neg_temp), 0)

        scores = torch.cat((pos_scores, neg_scores), 0)
        pos_probs = F.softmax(pos_scores, dim=1)[:, 1]
        neg_probs = F.softmax(neg_scores, dim=1)[:, 1]

        if pos_neg_labels is None:
            return None, pos_probs, neg_probs
        return self.xent(scores, pos_neg_labels), pos_probs, neg_probs

    def set_ngh_finder(self, ngh_finder):
        #self.ngh_finder = ngh_finder
        for agg in self.aggs:
            agg.set_ngh_finder(ngh_finder)
