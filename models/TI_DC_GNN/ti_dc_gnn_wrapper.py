from model_wrapper import ModelWrapper
import evaluation
import torch
import time
import os
import numpy as np
from .model.ti_dc_gnn import TiDcGNN
from models.TI_DC_GNN.utils.utils import EarlyStopMonitor, get_all_edges
from models.TI_DC_GNN.layers.mlp import MLP
from models.TI_DC_GNN.utils.graph_utils import get_neighbor_finder
from collections import defaultdict

from .train.pretrain import PreTrainer


class TiDcGNNWrapper(ModelWrapper):
    default_arg_config = {
        'data_mode': 'transaction',
        'data': 'wikipedia',
        'model_config': {
            'edge_memory_config': {
                'memory_dim': 100,
                'init_name': 'nodes',
                'init_fusion_fn_name': 'concat',
            },
            'node_memory_config': {  # always initialized as zero vector
                'use_memory': False,
                'memory_dim': 100
            },
            'local_mp_config': {
                'use_mp': True,
                'causal_config': {
                    'time_encoding_config': {'dim': 100},
                    'source_memory_readout_config': {
                        'fusion_fn_config': { 'fn_name': 'First', 'act_fn_name': 'Id' },
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None,},
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None,},
                    },
                    'edge_memory_readout_config': {
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'message_builder_config': {
                        'node_time_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'},
                        'edge_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'}
                    },
                    'dest_memory_readout_config': {
                        'same_as_source_readout': True,
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'message_aggregator_config': {
                        'target_time_fusion_config': {'fn_name': 'minus', 'act_fn_name': 'Id'},
                        'aggregator_config': {'fn_name': 'attn', 'act_fn_name': 'LeakyRelU', 'output_dim': None},
                    },
                    'node_memory_updater_config': {
                       'fusion_fn_config': {'fn_name': 'GRU', 'act_fn_name': 'Id'}
                    },
                    'dest_memory_write_in_fn_config': {
                        'fusion_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id'},
                    },
                },
                'conseq_config': {
                    'use_conseq': True,
                    'same_as_causal': True,
                    'time_encoding_config': {'dim': 100, 'share_with_causal': True},
                    'share_causal_weights': False,
                    'source_memory_readout_config': {
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'edge_memory_readout_config': {
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'message_builder_config': {
                        'node_time_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'},
                        'edge_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'}
                    },
                    'dest_memory_readout_config': {
                        'same_as_source_readout': True,
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'message_aggregator_config': {
                        'target_time_fusion_config': {'fn_name': 'minus', 'act_fn_name': 'Id'},
                        'aggregator_config': {'fn_name': 'attn', 'act_fn_name': 'LeakyRelU', 'output_dim': None},
                    },
                    'node_memory_updater_config': {
                       'fusion_fn_config': {'fn_name': 'GRU', 'act_fn_name': 'Id'}
                    },
                    'dest_memory_write_in_fn_config': {
                        'fusion_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id'},
                    },
                },
                'n_causal_steps': 1,
                'fusion_fn_config': {'fn_name': 'mean', 'act_fn_name': 'Id'},
                'edge_memory_update_config': {
                    'period_type': 'after_both',  # every, after_both, after_conseq
                    'node_memory_readout_config': {
                        'share_weights': False,
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'dest_edge_memory_readout_config': {
                        'share_weights': False,
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'message_builder_config': {
                        'share_weights': False,
                        'node_time_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'},
                        'edge_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'}
                    },
                    'message_aggregator_config': {
                        'share_weights': False,
                        'target_time_fusion_config': {'fn_name': 'minus', 'act_fn_name': 'Id'},
                        'aggregator_config': {'fn_name': 'attn', 'act_fn_name': 'LeakyRelU', 'output_dim': None},
                    },
                    'node_memory_updater_config': {
                        'share_weights': False,
                       'fusion_fn_config': {'fn_name': 'GRU', 'act_fn_name': 'Id'}
                    },
                    'dest_memory_write_in_fn_config': {
                        'share_weights': False,
                        'fusion_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id'},
                    },
                }
            },
            'global_mp_config': {
                'use_mp': False,
                'causal_config': {
                    'time_encoding_config': {'dim': 100, 'share_with_local': True},
                    'source_memory_readout_config': {
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'edge_memory_readout_config': {
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'message_builder_config': {
                        'node_time_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'},
                        'edge_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'}
                    },
                    'dest_memory_readout_config': {
                        'same_as_source_readout': True,
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'message_aggregator_config': {
                        'target_time_fusion_config': {'fn_name': 'minus', 'act_fn_name': 'Id'},
                        'aggregator_config': {'fn_name': 'attn', 'act_fn_name': 'LeakyRelU', 'output_dim': None},
                    },
                    'node_memory_updater_config': {
                        'fusion_fn_config': {'fn_name': 'GRU', 'act_fn_name': 'Id'}
                    },
                    'dest_memory_write_in_fn_config': {
                        'fusion_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id'},
                    },
                },
                'conseq_config': {
                    'use_conseq': True,
                    'same_as_causal': True,
                    'share_causal_weights': False,
                    'time_encoding_config': {'dim': 100, 'share_with_local': True},
                    'source_memory_readout_config': {
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'edge_memory_readout_config': {
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'message_builder_config': {
                        'node_time_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'},
                        'edge_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'}
                    },
                    'dest_memory_readout_config': {
                        'same_as_source_readout': True,
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'message_aggregator_config': {
                        'target_time_fusion_config': {'fn_name': 'minus', 'act_fn_name': 'Id'},
                        'aggregator_config': {'fn_name': 'attn', 'act_fn_name': 'LeakyRelU', 'output_dim': None},
                    },
                    'node_memory_updater_config': {
                        'fusion_fn_config': {'fn_name': 'GRU', 'act_fn_name': 'Id'}
                    },
                    'dest_memory_write_in_fn_config': {
                        'fusion_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id'},
                    },
                },
                'n_causal_steps': 1,
                'fusion_fn_name': 'mean',
                'edge_memory_update_config': {
                    'period_type': 'after_both',  # every, after_both, after_conseq
                    'time_encoding_config': {'dim': 100, 'share_with_local': True},
                    'node_memory_readout_config': {
                        'share_weights': False,
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'dest_edge_memory_readout_config': {
                        'share_weights': False,
                        'fusion_fn_config': {'fn_name': 'First', 'act_fn_name': 'Id'},
                        'transform_memory_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                        'transform_feats_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id', 'output_dim': None, },
                    },
                    'message_builder_config': {
                        'share_weights': False,
                        'node_time_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'},
                        'edge_fusion_fn_config': {'fn_name': 'concat', 'act_fn_name': 'Id'}
                    },
                    'message_aggregator_config': {
                        'share_weights': False,
                        'target_time_fusion_config': {'fn_name': 'minus', 'act_fn_name': 'Id'},
                        'aggregator_config': {'fn_name': 'attn', 'act_fn_name': 'LeakyRelU', 'output_dim': None},
                    },
                    'node_memory_updater_config': {
                        'share_weights': False,
                        'fusion_fn_config': {'fn_name': 'GRU', 'act_fn_name': 'Id'}
                    },
                    'dest_memory_write_in_fn_config': {
                        'share_weights': False,
                        'fusion_fn_config': {'fn_name': 'Id', 'act_fn_name': 'Id'},
                    },
                }
            },
            'embedding_module_config': {
                'time_encoding_config': {'dim': 100, 'share_with_local': True},
            },

        },
        'pretrain_config': {
            'train_config': {
                'n_epoch': 10,
                'lr': 0.0003,
                'patience': 5,
                'backprop_after_layers': 1,
                'backprop_per_time': 1,
                'weight_decay': 0,
            },
            'graph_config': {
                'causal_adj_builder_config': {'max_node_predecessors': 30},
                'train_transaction_rate': None,
                'old_neighbors_cnt': 0,
                'predecessor_sampler_config': {
                    'sampler_names': ['fixed', 'degree'],
                    'sampling_fn_names': ['full', 'uniform', 'random', 'last'],
                    'min_n_sample': 10,
                    'max_n_sample': 10,
                },
            },
        },
        'ssl_train_params': {
            'n_epoch': 10,
            'lr': 0.0003,
            'patience': 5,
            'weight_decay': 0,
        },
        'node_clf_train_params': {
            'n_epoch': 10,
            'lr': 0.0005,
            'patience': 5,
            'weight_decay': 0,
        },
        'ti_edge_time_dimension': 64,
        'uniform_neighbors': False,
        'ti_n_neighbors': 3,
        'embedding_module': 'graph_attention',  # ["graph_attention", "graph_sum", "identity", "time"]
        'embedding_n_neighbors': 20,
        'embedding_n_heads': 2,
        'embedding_n_layers': 1,
        'embedding_dropout': 0.1,
        'message_dim': 100,
        'memory_dim': 100,
        'use_memory': True,

        'gpu': 0,
        'use_validation':True,
        'force_cpu':True
    }

    def __init__(self, config=None, model_id=None):
        self.config = {**self.default_arg_config}
        if not (config is None):
            self.new_config = config
            self.config = {**self.config, **self.new_config}
        if model_id is None:
            self.model_id = f'ti_gnn_at_{time.ctime()}'
        else:
            self.model_id = f'{model_id} at {time.ctime()}'
        self.model_id = self.model_id.replace(':', '_')
        self.logger = self.prepare_logger(self.model_id)

    def initialize_model(self, full_data, train_data, node_features, edge_features, batch_params, data_setting_mode):
        self.device = torch.device(f'cuda:{self.config["gpu"]}'
                                   if torch.cuda.is_available() and not self.config['force_cpu']
                                   else 'cpu')

        self.max_idx = max(full_data.unique_nodes)
        self.data_setting_mode = data_setting_mode
        self.train_ngh_finder = get_neighbor_finder(train_data, uniform=self.config['uniform_neighbors'],
                                                    max_node_idx=self.max_idx)
        self.full_ngh_finder = get_neighbor_finder(full_data, uniform=self.config['uniform_neighbors'])

        sources, destinations = get_all_edges(full_data.edge_table)
        self.model = TiDcGNN(
            node_features=node_features,
            edge_features=edge_features,
            device=self.device,
            sources=sources,
            destinations=destinations,
            edge_memory_config=self.config['model_config']['edge_memory_config'],
            node_memory_config=self.config['model_config']['node_memory_config'],
            local_mp_config=self.config['model_config']['local_mp_config'],
            global_mp_config=self.config['model_config']['global_mp_config'],
            embedding_module_config=self.config['model_config']['embedding_module'],
        )
        self.pretrainer = PreTrainer(self.config['pretrain_config'], self.model.local_mp, self.logger)


        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config['lr'],
                                          weight_decay=self.config['weight_decay'])
        self.batch_params = batch_params

    def compute_edge_probabilities(
            self,
            batch,
            negative,
            eval_mode,
            data_setting_mode,
            **model_params):

        """
        Predict 2 batches of edges - from source to destination and from source to negative samples.
        We separate real and sample edges, as some models (e.g. TGN) need to know, which data is real to utilize it afterwards.
        """
        (
            sources_batch,
            destinations_batch,
            timestamps_batch,
            edge_idxs_batch,
            _,
        ) = batch

        if eval_mode != 'train':
            try:
                current_memory = self.model.memory.backup_memory()
                self.model.memory.restore_memory(self.model.memory.val_memory_backup)
            except:
                pass
        real_probabilities, sampled_probabilities = self.model.compute_edge_probabilities(
            sources_batch,
            destinations_batch,
            negative,
            timestamps_batch,
            edge_idxs_batch,
            **model_params)
        if eval_mode != 'train':
            try:
                self.model.memory.restore_memory(current_memory)
            except:
                pass
        return real_probabilities, sampled_probabilities

    def compute_clf_probability(
            self,
            data,
            eval_mode,
            data_setting_mode,
            **kwargs):
        res_probs = []
        (
            sources_batch,
            destinations_batch,
            timestamps_batch,
            edge_idxs_batch,
            _
        ) = data
        if eval_mode != 'train':
            with torch.no_grad():
                self.decoder.eval()
                self.model.eval()

                source_embedding, destination_embedding, _ = self.model.compute_temporal_embeddings(
                    sources_batch,
                    destinations_batch,
                    destinations_batch,
                    timestamps_batch,
                    edge_idxs_batch,
                    kwargs['n_neighbors'])
        else:
            source_embedding, destination_embedding, _ = self.model.compute_temporal_embeddings(
                sources_batch,
                destinations_batch,
                destinations_batch,
                timestamps_batch,
                edge_idxs_batch,
                kwargs['n_neighbors'])
        pred_prob = self.decoder(source_embedding).sigmoid()
        return pred_prob

    def load_model(self, model_path):
        """
        Load model weights and return True, or finish initialization of model and return False
        """
        if not (model_path is None) and os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path))
            is_trained = True
        else:
            is_trained = False

        return is_trained

    def train_model(self, train_data, val_data, train_sampler, val_sampler, do_clf):
        pretrain_metrics = self.pretrainer.train(train_data, val_data)
        train_metrics_unsupervised = self.train_self_supervised(train_data, val_data, train_sampler, val_sampler)
        if do_clf:
            train_metrics_supervised = self.train_supervised(train_data, val_data, train_sampler, val_sampler)
        else:
            train_metrics_supervised = {}
        return {
            'pretrain_metrics': pretrain_metrics,
            'train_unsupervised': train_metrics_unsupervised,
            'train_supervised': train_metrics_supervised,
        }

    def train_self_supervised(self, train_data, val_data, train_sampler, val_sampler):
        num_instance = len(train_data.edge_table)

        self.logger.info('num of training instances: {}'.format(num_instance))

        train_metrics = defaultdict(list)

        self.early_stopper = EarlyStopMonitor(max_round=self.config['patience'])

        # run an epoch
        for epoch in range(self.config['n_epoch']):
            self.logger.info('start {} epoch'.format(epoch))
            # metrics = self.run_unsupervised_epoch(epoch_config, train_data, val_data )
            # keep_training = save_epoch_data(*metrics)
            # if not keep_training:
            #     print(f'We dont want to keep training, stopping on {epoch} epoch')

            start_epoch = time.time()
            # Training
            # Reinitialize memory of the model at the start of each epoch
            if self.config['use_memory']:
                self.model.memory.__init_memory__()
            # Train using only training graph
            self.model.set_neighbor_finder(self.train_ngh_finder)
            m_loss = []

            loss = 0
            self.optimizer.zero_grad()
            start = None
            training_data = list(train_data(**self.batch_params['train']))
            self.model = self.model.train()
            for i, batch in enumerate(training_data):
                # print(f'Batch {i}')
                if i % 100 == 0:
                    print(
                        f'{i}/{len(training_data)} batches passed ... It took {round(time.time() - start if start else 0, 2)} seconds')
                    start = time.time()
                (
                    sources_batch,
                    destinations_batch,
                    timestamps_batch,
                    edge_idxs_batch,
                    _,
                ) = batch
                size = len(sources_batch)
                _, negatives_batch = train_sampler.sample(size)
                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=self.device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=self.device)

                pos_prob, neg_prob = self.compute_edge_probabilities(
                    batch,
                    negatives_batch,
                    eval_mode='train',
                    data_setting_mode=self.data_setting_mode,
                    n_neighbors=self.config['n_degree']
                )

                loss += self.criterion(
                    pos_prob.reshape(pos_label.shape), pos_label
                ) + self.criterion(
                    neg_prob.reshape(neg_label.shape), neg_label
                )

                if (not i % self.config['backprop_every']) & (i):
                    loss /= self.config['backprop_every']

                    loss.backward()
                    self.optimizer.step()
                    m_loss.append(loss.item())
                    # Detach memory after 'BACKPROP_EVERY' number of batches so we don't backpropagate to
                    # the start of time
                    if self.config['use_memory']:
                        self.model.memory.detach_memory()

                    loss = 0
                    self.optimizer.zero_grad()

            epoch_time = time.time() - start_epoch
            train_metrics['epoch_times'].append(epoch_time)

            # Validation
            # Validation uses the full graph
            self.model.set_neighbor_finder(self.full_ngh_finder)

            if self.config['use_memory']:
                # Backup memory at the end of training, so later we can restore it and use it for the
                # validation on unseen nodes
                train_memory_backup = self.model.memory.backup_memory()
            torch.cuda.empty_cache()
            self.model = self.model.eval()
            transductive_val = evaluation.eval_edge_prediction(model=self,
                                                               data=val_data,
                                                               negative_edge_sampler=val_sampler,
                                                               data_setting_mode='transductive',
                                                               batch_params=self.batch_params['val'],
                                                               eval_mode='val',
                                                               n_neighbors=self.config['n_degree'],
                                                               )
            if self.config['use_memory']:
                val_memory_backup = self.model.memory.backup_memory()
                # Restore memory we had at the end of training to be used when validating on new nodes.
                # Also backup memory after validation so it can be used for testing (since test edges are
                # strictly later in time than validation edges)
                self.model.memory.restore_memory(train_memory_backup)

            if self.config['use_memory']:
                # Restore memory we had at the end of validation
                self.model.memory.restore_memory(val_memory_backup)
                self.model.memory.val_memory_backup = val_memory_backup
            total_epoch_time = time.time() - start_epoch

            train_metrics['val_aps'].append(transductive_val['AP'])
            train_metrics['train_losses'].append(np.mean(m_loss))

            # train_metrics['total_epoch_times'].append(total_epoch_time)

            self.logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            self.logger.info(f'Epoch mean loss: {np.mean(m_loss)}')
            self.logger.info(f'val auc: {transductive_val["AUC"]}')
            self.logger.info(f'val ap: {transductive_val["AP"]}  ')

            # Early stopping
            if self.early_stopper.early_stop_check(transductive_val['AP']):
                self.logger.info(f'No improvement over {self.early_stopper.max_round} epochs, stop training')
                self.logger.info(f'Loading the best model at epoch {self.early_stopper.best_epoch}')
                best_model_path = self.get_checkpoint_path(self.early_stopper.best_epoch)
                self.model.load_state_dict(torch.load(best_model_path))
                torch.save(self.model.state_dict(), self.get_checkpoint_path(-1, 'graph'))
                self.logger.info(f'Loaded the best model at epoch {self.early_stopper.best_epoch} for inference')
                self.model.eval()
                break
            elif (epoch + 1 == self.config['n_epoch']):
                self.model.eval()

            torch.save(self.model.state_dict(), self.get_checkpoint_path(epoch, 'graph'))

        return train_metrics

    def train_supervised(self, train_data, val_data, train_sampler, val_sampler):
        num_instance = train_data.edge_table.shape[0]

        self.logger.debug('Num of training instances: {}'.format(num_instance))

        self.model.eval()
        self.logger.info('TGN models loaded')
        self.logger.info('Start training node classification task')

        self.decoder = self.get_decoder(self.model.node_raw_features)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.config['lr_decoder'])
        self.decoder = self.decoder.to(self.device)
        self.decoder_loss_criterion = torch.nn.BCELoss()

        train_metrics = defaultdict(list)

        early_stopper = EarlyStopMonitor(max_round=self.config['patience'])
        for epoch in range(self.config['n_epoch']):
            start_epoch = time.time()

            # Initialize memory of the model at each epoch
            if self.config['use_memory']:
                self.model.memory.__init_memory__()

            self.model = self.model.eval()
            self.decoder = self.decoder.train()
            loss = 0

            for i, batch in enumerate(train_data(**self.batch_params['train'])):
                (
                    sources_batch,
                    destinations_batch,
                    timestamps_batch,
                    edge_idxs_batch,
                    labels_batch
                ) = batch

                size = len(sources_batch)

                self.decoder_optimizer.zero_grad()
                with torch.no_grad():
                    source_embedding, destination_embedding, _ = self.model.compute_temporal_embeddings(sources_batch,
                                                                                                        destinations_batch,
                                                                                                        destinations_batch,
                                                                                                        timestamps_batch,
                                                                                                        edge_idxs_batch,
                                                                                                        self.config[
                                                                                                            'n_degree'])

                labels_batch_torch = torch.from_numpy(
                    labels_batch).float().to(self.device)
                pred = self.decoder(source_embedding).sigmoid()
                decoder_loss = self.decoder_loss_criterion(pred, labels_batch_torch)
                decoder_loss.backward()
                self.decoder_optimizer.step()
                loss += decoder_loss.item()
            train_metrics['train_losses'].append(loss / i + 1)

            val_auc = evaluation.eval_node_bin_classification(
                model=self,
                data=val_data,
                data_setting_mode='transductive',
                batch_params=self.batch_params['val'],
                eval_mode='val',
                n_neighbors=self.config['n_degree'])

            train_metrics['val_aucs'].append(val_auc['AUC ROC'])
            train_metrics['epoch_times'].append(time.time() - start_epoch)

            self.logger.info(
                f'Epoch {epoch}: train loss: {train_metrics["train_losses"][-1]}, val auc: {train_metrics["val_aucs"][-1]}, time: {train_metrics["epoch_times"][-1]}')

            if early_stopper.early_stop_check(val_auc['AUC ROC']):
                self.logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                break
            else:
                torch.save(self.decoder.state_dict(), self.get_checkpoint_path(epoch, 'decoder'))
        self.logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_model_path = self.get_checkpoint_path(early_stopper.best_epoch, 'decoder')
        self.decoder.load_state_dict(torch.load(best_model_path))
        torch.save(self.decoder.state_dict(), self.get_checkpoint_path(-1, 'decoder'))
        self.logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        self.decoder.eval()
        return train_metrics

    def get_decoder(self, node_features):
        return MLP(node_features.shape[1], drop=self.config['drop_out'])

    def get_inference_params(self):
        return {'n_neighbors': self.config['n_degree']}

    def get_results_path(self, ):
        return os.path.join(f'./logs/{self.model_id}.pkl')

    def get_checkpoint_path(self, epoch, part='graph', final=False):
        """
        If epoch<0, the training is done and we want path to final model
        """

        if not os.path.isdir('./saved_checkpoints'):
            os.mkdir('./saved_checkpoints')
        if epoch < 0:
            return f'./saved_checkpoints/{self.model_id}-{part}-final.pth'
        else:
            return f'./saved_checkpoints/{self.model_id}-{part}-{epoch}.pth'


