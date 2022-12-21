from collections import defaultdict
import time

import numpy as np
import torch

from models.TI_DC_GNN.ti_dual_graph.pretrain.local_graph_for_pretrain import TiLocalGraphBuilder
from models.TI_GNN_new.ti_dual_graph.sampler import get_directed_edge_sampler
from models.TI_GNN_new.ti_dual_graph.ti_graph_builder import TiDualAdjListBuilder
from models.TI_GNN_new.utils.utils import EarlyStopMonitor

from models.TI_DC_GNN.graph.sampler import get_predecessor_samplers
from models.TI_DC_GNN.graph.ti_local_graph import LocalGraphBuilder
from models.TI_DC_GNN.modules.mail_box import MailBox
from models.TI_DC_GNN.modules.pretrain.local_mp_pretrain_model import LocalMPModelPretrain
from models.TI_DC_GNN.utils.utils import flatten_list


class PreTrainer:
    def __init__(self, train_config: dict, graph_config, loss_config, logger, device,
                 local_mp_model: LocalMPModelPretrain, neighbor_finder,
                 feature_properties, memory_properties, time_encoding):
        self.local_mp_model = local_mp_model
        self.logger = logger
        self.device = device
        self.train_config = train_config
        self.loss_config = loss_config
        self.graph_config = graph_config
        self.optimizer = torch.optim.Adam(self.local_mp_model.parameters(),
                                          lr=self.train_config['lr'],
                                          weight_decay=self.train_config['weight_decay'])
        self.mailbox = MailBox(neighbor_finder=neighbor_finder,
                               old_neighbors_cnt=graph_config['old_neighbors_cnt'],
                               causal_adj_builder_config=graph_config['causal_adj_builder_config'])
        predecessor_samplers = get_predecessor_samplers(self.graph_config['predecessor_sampler_config'])
        self.local_graph_builder = LocalGraphBuilder(predecessor_samplers=predecessor_samplers,
                                                     sample_once=self.graph_config['sample_once'])
        self.feature_properties = feature_properties
        self.memory_properties = memory_properties
        self.time_encoding = time_encoding
        self.bce_loss = torch.nn.BCELoss()

    def loss(self, grouped_sources, grouped_edges, destinations):
        l_asym = 0
        if self.loss_config['use_asym']:
            pass

    def node_asym_loss(self, sources_ids, destinations_ids):
        positive_scores = self.local_mp_model.decoder_node_similarity(self.local_mp_model.edge_memory[sources_ids],
                                                                      self.local_mp_model.edge_memory[destinations_ids])
        negative_scores = self.local_mp_model.decoder_node_similarity(sources_ids, destinations_ids)
        labels = torch.cat([torch.ones(positive_scores.shape[0]), torch.zeros(negative_scores.shape[0])])
        return self.bce_loss(torch.cat([positive_scores, negative_scores]), labels)

    def node_sim_contrastive_loss(self):
        pass

    def node_edge_sim_loss(self, positive_triplets, negative_triplets):
        positive_sources_ids, positive_destinations_ids, positive_edge_ids = positive_triplets
        negative_sources_ids, negative_destinations_ids, negative_edge_ids = negative_triplets

        positive_scores = self.local_mp_model.decoder_node_edge_similarity(source_memory_batch=self.local_mp_model.edge_memory[positive_sources_ids],
                                                                           edge_memory_batch=self.local_mp_model.edge_memory[positive_sources_ids],
                                                                           dest_memory_batch=self.local_mp_model.edge_memory[positive_destinations_ids])

    def construct_dual_local_graph(self, data):
        if self.graph_config['train_transaction_rate'] is None:
            edge_table = data.edge_table
        else:
            transactions_cnt = np.round(self.graph_config['train_transaction_rate'] * len(data.edge_table))
            edge_table = data.edge_table[:transactions_cnt, :]

        edge_idxs, timestamps, sources, destinations = \
            tuple(edge_table[:, i] for i in range(4))
        self.mailbox.add_batch(sources, destinations, timestamps, edge_idxs)
        start_interval_time = self.mailbox.get_start_interval_time()
        causal_in_adj_list = self.mailbox.get_causal_adj_list(start_interval_time)
        return self.local_graph_builder.build(causal_in_adj_list)

    def train(self, train_data, val_data):
        num_instance = len(train_data.edge_table)
        self.logger.info('--------PRETRAINING PHASE STARTED--------')
        self.logger.info('num of training instances: {}'.format(num_instance))
        train_metrics = defaultdict(list)
        self.early_stopper = EarlyStopMonitor(max_round=self.train_config['patience'])
        local_dual_graph_full = self.construct_dual_local_graph(train_data)
        for epoch in range(self.train_config['n_epoch']):
            self.logger.info('start {} epoch'.format(epoch))
            start_epoch_time = time.time()
            mean_loss = self.run_train_epoch(local_dual_graph_full)
            end_train_epoch_time = time.time()
            total_epoch_time = end_train_epoch_time - start_epoch_time
            train_metrics['total_epoch_times'].append(total_epoch_time)
            self.logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            self.logger.info(f'Epoch mean loss: {mean_loss}')

    def run_train_epoch(self, local_dual_graph):
        loss_lst = []
        conseq_list_node_edge_ids = []
        n_layers = len(local_dual_graph.layer_to_node_ids)
        for layer_num in range(n_layers):
            for _ in range(self.config['backprop_per_time']):

            dest_node_ids = local_dual_graph.layer_to_node_ids[layer_num]
            grouped_source_ids, grouped_edge_ids = local_dual_graph.get_nodes_predecessors(dest_node_ids)
            source_ids_flatten = torch.LongTensor(flatten_list(grouped_source_ids))
            edge_ids_flatten = torch.LongTensor(flatten_list(grouped_edge_ids))
            dest_memory, dest_node_ids_torch = self.local_mp_model.forward_causal(local_dual_graph=local_dual_graph,
                                                                          feature_properties=self.feature_properties,
                                                                          memory_properties=self.memory_properties,
                                                                          time_encoding=self.time_encoding,
                                                                          layer_num=layer_num)

            cur_node_features = self.dual_node_features[nodes_in_layers]
            cur_node_memory = self.dual_node_memory[nodes_in_layers]
            cur_edge_features = self.dual_edge_features[nodes_in_layers]
            for _ in range(self.config['backprop_per_time']):
                cur_node_memory = self.model(cur_node_memory,
                                             cur_node_features,
                                             cur_edge_features,
                                             train_dual_graph_full.causal)
                if self.config['use_conseq']:
                    cur_node_memory = self.model(cur_node_memory,
                                                 cur_node_features,
                                                 cur_edge_features,
                                                 train_dual_graph_full.conseq)
            loss = self.loss(cur_node_memory, train_dual_graph_full)
            loss /= self.config['backprop_per_time']
            loss.backward()
            self.optimizer.step()
            loss_lst.append(loss.item())
            self.optimizer.zero_grad()
        if self.n_causal_steps is not None and self.use_conseq:
            causal_node_memory = node_edge_properties.node_memory
            node_edge_properties = self.apply_graph_conseq_mp(conseq_list_node_edge_ids,
                                                              node_edge_properties,
                                                              time_encoding)
            node_edge_properties.node_memory = self.appply_fusion_fn(causal_node_memory,
                                                                     node_edge_properties.node_memory)
        return np.mean(loss_lst)
