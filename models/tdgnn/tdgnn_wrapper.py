from model_wrapper import ModelWrapper
from data_tools.data_processing import compute_time_statistics
import evaluation
import pickle
import torch
import time
import os
import math
import numpy as np
from .model.tdgnn import TDGNN_GraphSage
from .utils import EarlyStopMonitor, MLP, get_neighbor_finder
from collections import defaultdict


class TDGNNWrapper(ModelWrapper):
    default_arg_config = {
        'layer_num': 2,
        'embed_dim': 128,
        'n_layer': 2,
        'edge_agg_name': 'mean', #possible: mean, hadamard, weight-l1, activation, original
        'num_classes': 2,
        'data_mode': 'transaction',
        'num_sample': None,
        'bs': 200,
        'n_epoch': 20,
        'lr': 0.7,
        'drop_out': 0.2,
        'patience': 5,
        'neighbour_max': None,
        'gpu': 0,
        'use_validation': True,
        'force_cpu': False
    }

    def __init__(self, config=None, model_id=None):
        super(TDGNNWrapper, self).__init__()
        self.config = {**self.default_arg_config}
        if not (config is None):
            self.new_config = config
            self.config = {**self.config, **self.new_config}
        if model_id is None:
            self.model_id = f'tdgnn_at_{time.ctime()}'.replace(':','_')
        else:
            self.model_id = f'{model_id} at {time.ctime()}'.replace(':','_')
        self.logger = self.prepare_logger(self.model_id)

    def initialize_model(self, full_data, train_data, node_features, edge_features, batch_params, data_setting_mode):
        self.device = f'cuda:{self.config["gpu"]}' if torch.cuda.is_available() and not self.config[
            'force_cpu'] else 'cpu'


        self.max_idx = max(full_data.unique_nodes)
        self.train_ngh_finder = get_neighbor_finder(
            train_data,
            max_node_idx=self.max_idx,
            neighbour_max=self.config['neighbour_max'],
        )

        self.full_ngh_finder = get_neighbor_finder(
            full_data,
            neighbour_max=self.config['neighbour_max']
        )

        self.model = TDGNN_GraphSage(
            feat_data=node_features,
            ngh_finder=self.train_ngh_finder,
            device=self.device,
            layer_num=self.config['n_layer'],
            embed_dim=self.config['embed_dim'],
            edge_agg_name=self.config['edge_agg_name'],
            num_classes=self.config['num_classes'],
            num_sample=self.config['num_sample']
        )
        self.model.to(self.device)
        #self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config['lr'])
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
            _,
            _,
        ) = batch

        _, real_probabilities, sampled_probabilities = self.model.compute_edge_probabilities(
            sources_batch,
            destinations_batch,
            negative,
            timestamps_batch,
        )
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
            _,
            _
        ) = data
        if eval_mode!='train':
            with torch.no_grad():
                self.decoder.eval()
                self.model.eval()

        source_embedding, _ = self.model.compute_temporal_embeddings(
            sources_batch,
            destinations_batch,
            timestamps_batch,
        )
        pred_prob = self.decoder(source_embedding)[:, 1]
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
        train_metrics_1 = self.train_self_supervised(train_data, val_data, train_sampler, val_sampler)
        train_metrics_2 = self.train_supervised(train_data, val_data, train_sampler, val_sampler)
        return {'train_unsupervised': train_metrics_1, 'train_supervised': train_metrics_2}

    def train_self_supervised(self, train_data, val_data, train_sampler, val_sampler):
        num_instance = len(train_data.edge_table)
        num_batch = math.ceil(num_instance / self.config['bs'])

        self.logger.info('num of training instances: {}'.format(num_instance))
        self.logger.info('num of batches per epoch: {}'.format(num_batch))

        train_metrics = defaultdict(list)
        self.early_stopper = EarlyStopMonitor(max_round=self.config['patience'])

        # run an epoch
        for epoch in range(self.config['n_epoch']):
            self.logger.info('start {} epoch'.format(epoch))
            self.model.set_ngh_finder(self.train_ngh_finder)
            start_epoch = time.time()
            m_loss = []
            self.model = self.model.train()
            for i, batch in enumerate(train_data(**self.batch_params['train'])):

                (
                    sources_batch,
                    destinations_batch,
                    timestamps_batch,
                    edge_idxs_batch,
                    _,
                ) = batch
                size = len(sources_batch)
                _, negatives_batch = train_sampler.sample(size)
                pos_label = torch.ones(size)
                neg_label = torch.zeros(size)
                pos_neg_labels = torch.cat((pos_label, neg_label), 0).long()
                #print(pos_neg_labels)
                self.optimizer.zero_grad()
                #print('before comp-n')
                #print('number of samples in batch:', len(pos_neg_labels))
                loss_value, _, _ = self.model.compute_edge_probabilities(
                    sources_batch, destinations_batch, negatives_batch, timestamps_batch, pos_neg_labels
                )
                #print('after comp-n')
                loss_value.backward()
                self.optimizer.step()
                m_loss.append(loss_value.item())

            epoch_time = time.time() - start_epoch
            train_metrics['epoch_times'].append(epoch_time)

            # Validation

            torch.cuda.empty_cache()
            self.model.set_ngh_finder(self.full_ngh_finder)
            self.model = self.model.eval()
            transductive_val = evaluation.eval_edge_prediction(
                model=self,
                data=val_data,
                negative_edge_sampler=val_sampler,
                data_setting_mode='transductive',
                batch_params=self.batch_params['val'],
                eval_mode='val',
            )

            train_metrics['val_aps'].append(transductive_val['AP'])
            train_metrics['train_losses'].append(np.mean(m_loss))

            # Save temporary results to disk
            pickle.dump(train_metrics, open(self.get_results_path(), "wb"))

            total_epoch_time = time.time() - start_epoch
            train_metrics['total_epoch_times'].append(total_epoch_time)

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
        num_batch = math.ceil(num_instance / self.config['bs'])

        self.logger.debug('Num of training instances: {}'.format(num_instance))

        self.model.eval()
        self.logger.info('TGN models loaded')
        self.logger.info('Start training node classification task')

        self.decoder = self.get_decoder(self.config['embed_dim'])
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        self.decoder = self.decoder.to(self.device)
        decoder_loss_criterion = torch.nn.BCELoss()

        train_metrics = defaultdict(list)

        early_stopper = EarlyStopMonitor(max_round=self.config['patience'])
        for epoch in range(self.config['n_epoch']):
            start_epoch = time.time()

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

                decoder_optimizer.zero_grad()
                with torch.no_grad():
                    source_embedding, destination_embedding = self.model.compute_temporal_embeddings(sources_batch,
                                                                                                        destinations_batch,
                                                                                                        timestamps_batch)
                #print(source_embedding.shape)
                #print(self.model.features.shape)
                labels_batch_torch = torch.from_numpy(
                    labels_batch).float().to(self.device)
                pred = self.decoder(source_embedding)[:, 1]
                decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
                decoder_loss.backward()
                decoder_optimizer.step()
                loss += decoder_loss.item()
            train_metrics['train_losses'].append(loss / num_batch)

            val_auc = evaluation.eval_node_bin_classification(
                model=self,
                data=val_data,
                data_setting_mode='transductive',
                batch_params=self.batch_params['val'],
                eval_mode='val',
            )

            train_metrics['val_aucs'].append(val_auc['AUC ROC'])
            train_metrics['epoch_times'].append(time.time() - start_epoch)
            pickle.dump(train_metrics, open(self.get_results_path(), "wb"))

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
        pickle.dump(train_metrics, open(self.get_results_path(), "wb"))
        return train_metrics

    def get_inference_params(self):
        return {}

    def get_decoder(self, embed_dim):
        return MLP(embed_dim, drop=self.config['drop_out'])

    def get_results_path(self):
        return os.path.join(f'./logs/{self.model_id}.pkl')

    def get_checkpoint_path(self, epoch, part='graph', final=False):
        """
        If epoch<0, the training is done and we want path to final model
        """

        if not os.path.isdir('./saved_checkpoints'):
            os.mkdir('./saved_checkpoints')
        if epoch < 0:
            return f'./saved_checkpoints/TGN-{self.model_id}-{part}-final.pth'
        else:
            return f'./saved_checkpoints/TGN-{self.model_id}-{part}-{epoch}.pth'
