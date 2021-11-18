import time
from collections import defaultdict
import math
import torch

from data_tools.data_interface import GraphContainer
from model_wrapper import ModelWrapper
from models.HiLi.model import Model
from models.tgn.utils import get_neighbor_finder, EarlyStopMonitor, MLP
import torch.nn.functional as F
import evaluation
import numpy as np
import os


class HiliWrapper(ModelWrapper):
    default_arg_config = {
        "data": "wikipedia",
        "emb_dim": 128,
        "size": 3,
        "item_max": 3,
        "item_pow": 0.75,
        "user_max": 4,
        "user_pow": 0.75,
        "uniform": True,
        "lr": 3e-4,
        "lr_decoder": 3e-4,
        "n_epoch": 1,
        'patience':5,
        "drop_out": 0.1,
        'n_degree':10,
    }

    def __init__(self, config=None, model_id=None):
        self.config = {**self.default_arg_config}
        if not (config is None):
            self.new_config = config
            self.config = {**self.config, **self.new_config}
        if model_id is None:
            self.model_id = f'hili_at_{time.ctime()}'
        else:
            self.model_id = f'{model_id} at {time.ctime()}'
        self.model_id = self.model_id.replace(':', '_')
        self.logger = self.prepare_logger(self.model_id)

        # self.history = {
        #     'user_history': defaultdict(lambda: Queue(self.default_arg_config['size'])),
        #     'item_history': defaultdict(lambda: Queue(self.default_arg_config['size'])),
        # }

    def initialize_model(self, full_data: GraphContainer, train_data, node_features, edge_features, batch_params,
                         data_setting_mode='transductive'):
        self.device = f'cuda:{self.config["gpu"]}' if torch.cuda.is_available() and not self.config[
            'force_cpu'] else 'cpu'

        self.edge_features = edge_features
        self.num_feats = edge_features.edge_table.shape[1]
        self.num_users = len(set(full_data.edge_table[:, 2]))
        self.num_items = len(set(full_data.edge_table[:, 3]))
        
        self.not_bipartite = self.config["data"] not in ["wikipedia", "reddit"]
        self.eth_flg = "eth" in self.config["data"]
        if self.not_bipartite:
            self.num_users = full_data.n_unique_nodes
            self.num_items = full_data.n_unique_nodes
            
        self.model = Model(self.config["emb_dim"], self.config["emb_dim"], self.num_users, self.num_items, self.num_feats, self.config["size"])
        self.model = self.model.to(self.device)

        self.max_idx = max(full_data.unique_nodes)

        self.train_ngh_finder = get_neighbor_finder(train_data, uniform=self.config['uniform'],
                                                    max_node_idx=self.max_idx)
        self.full_ngh_finder = get_neighbor_finder(full_data, uniform=self.config['uniform'])

        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.batch_params = batch_params

        self.dyn_user_emb = F.normalize(torch.rand((self.num_users, self.config["emb_dim"])), dim=1)
        self.stat_user_emb = torch.eye(self.num_users)
        if not self.eth_flg:
            self.dyn_item_emb = F.normalize(torch.rand((self.num_items, self.config["emb_dim"])), dim=1)
            self.stat_item_emb = torch.eye(self.num_items) * self.config["item_max"]

    def compute_clf_probability(
        self, 
        data, 
        eval_mode, 
        data_setting_mode,
        **kwargs):
        self.decoder.eval()
        self.model.eval()
        self.process_batch(data)
        return self.decoder(self.dyn_user_emb[data[0]]).sigmoid()
    
    def compute_edge_probabilities(self, batch, negatives, eval_mode, data_setting_mode, **model_params):
        _, pos, neg = self.process_batch(batch, negatives)
        return -pos, -neg

    def load_model(self, model_path):
        if not (model_path is None) and os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path))
            is_trained = True
        else:
            is_trained = False
            
        return is_trained
    
    def train_model(self, train_data, val_data, train_sampler, val_sampler, do_clf):
    
        train_metrics_1 = self.train_self_supervised(train_data, val_data, train_sampler, val_sampler  )
        if do_clf:
            train_metrics_2 = self.train_supervised(train_data, val_data, train_sampler, val_sampler)
        else:
            train_metrics_2 = {}
        return {'train_unsupervised':train_metrics_1, 'train_supervised':train_metrics_2}

    def train_supervised(self, train_data, val_data, train_sampler, val_sampler):
        num_instance = train_data.edge_table.shape[0]
        # num_batch = math.ceil(num_instance / self.batch_params['train']["batch_size"])

        self.logger.debug('Num of training instances: {}'.format(num_instance))

        self.model.eval()
        self.logger.info('TGN models loaded')
        self.logger.info('Start training node classification task')

        self.decoder = self.get_decoder(self.config["emb_dim"])
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.config['lr_decoder'])
        self.decoder = self.decoder.to(self.device)
        self.decoder_loss_criterion = torch.nn.BCELoss()
        
        train_metrics = defaultdict(list)
        
        early_stopper = EarlyStopMonitor(max_round=self.config['patience'])
        for epoch in range(self.config['n_epoch']):
            start_epoch = time.time()
            self.model = self.model.train()
            self.decoder = self.decoder.train()
            
            self.dyn_user_emb = F.normalize(torch.rand((self.num_users, self.config["emb_dim"])), dim=1)
            self.stat_user_emb = torch.eye(self.num_users)
            if not self.eth_flg:
                self.dyn_item_emb = F.normalize(torch.rand((self.num_items, self.config["emb_dim"])), dim=1)
                self.stat_item_emb = torch.eye(self.num_items) * self.config["item_max"]
            
            loss = 0
            for i, batch in enumerate(train_data(**self.batch_params['train'])):
                self.decoder_optimizer.zero_grad()
                enc_loss, _, _ = self.process_batch(batch)
                labels_batch_torch = torch.from_numpy(batch[4]).float().to(self.device)
                pred = self.decoder(self.dyn_user_emb[batch[0]]).sigmoid()
                decoder_loss = self.decoder_loss_criterion(pred, labels_batch_torch) + enc_loss
                decoder_loss.backward()
                self.decoder_optimizer.step()
                loss += decoder_loss.item()
            train_metrics['train_losses'].append(loss / i+1)
            
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
        
        
    
    
    def train_self_supervised(self, train_data, val_data, train_sampler, val_sampler):
        num_instance = len(train_data.edge_table)
        # num_batch = math.ceil(num_instance / self.batch_params['train']["batch_size"])

        self.logger.info('num of training instances: {}'.format(num_instance))
        # self.logger.info('num of batches per epoch: {}'.format(num_batch))

        train_metrics = defaultdict(list)
        self.early_stopper = EarlyStopMonitor(max_round=self.config['patience'])

        for epoch in range(self.config['n_epoch']):
            self.logger.info('start {} epoch'.format(epoch))
            start_epoch = time.time()
            m_loss = []
            
            start = None
            training_data = train_data(**self.batch_params['train'])

            self.dyn_user_emb = F.normalize(torch.rand((self.num_users, self.config["emb_dim"])), dim=1)
            self.stat_user_emb = torch.eye(self.num_users)
            if not self.eth_flg:
                self.dyn_item_emb = F.normalize(torch.rand((self.num_items, self.config["emb_dim"])), dim=1)
                self.stat_item_emb = torch.eye(self.num_items) * self.config["item_max"]

            for i, batch in enumerate(training_data):
                self.optimizer.zero_grad()
                if i % 100 == 0:
                    print(f'{i}/{training_data.num_batches} batches passed ... It took {round(time.time() - start if start else 0, 2)} seconds')
                    start = time.time()
                
                self.model = self.model.train()
                
                loss, _, _ = self.process_batch(batch)

                loss.backward()
                self.optimizer.step()
                m_loss.append(loss.item())
                
            epoch_time = time.time() - start_epoch
            train_metrics['epoch_times'].append(epoch_time)
            
            transductive_val = evaluation.eval_edge_prediction(model=self,
                                                            data=val_data,
                                                            negative_edge_sampler=val_sampler,
                                                            data_setting_mode='transductive',
                                                            batch_params=self.batch_params['val'],
                                                            eval_mode='val',
                                                            n_neighbors=self.config['n_degree'],
                                                            )
            
            train_metrics['val_aps'].append(transductive_val['AP'])
            train_metrics['train_losses'].append(np.mean(m_loss))
            
            total_epoch_time = time.time() - start_epoch
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
            elif (epoch+1==self.config['n_epoch']):
                self.model.eval()
                
            torch.save(self.model.state_dict(), self.get_checkpoint_path(epoch, 'graph'))
        return train_metrics

    def process_batch(self, batch, negatives=None):
        (
            sources_batch,
            destinations_batch,
            timestamps_batch,
            edge_idxs_batch,
            labels,
        ) = batch
        
        self.dyn_user_emb.detach_()
        if not self.eth_flg:
            self.dyn_item_emb.detach_()
        
        loss = 0
        
        
        sources_batch = torch.Tensor(sources_batch).type(torch.LongTensor)
        destinations_batch = torch.Tensor(destinations_batch).type(torch.LongTensor)
        timestamps_batch = torch.Tensor(timestamps_batch)

        src_neighbors, src_edge_idxs, src_edge_times = (self.full_ngh_finder if negatives is not None else self.train_ngh_finder).get_temporal_neighbor(
            sources_batch,timestamps_batch, n_neighbors=self.config["size"])
        
        src_neighbors = torch.Tensor(src_neighbors).type(torch.LongTensor)
        src_edge_times = torch.Tensor(src_edge_times)
        
        dst_neighbors, dst_edge_idxs, dst_edge_times = (self.full_ngh_finder if negatives is not None else self.train_ngh_finder).get_temporal_neighbor(
            destinations_batch, timestamps_batch, n_neighbors=self.config["size"])
        dst_edge_times = torch.Tensor(dst_edge_times)

        src_neighbors = torch.where(src_neighbors == 0, 0, src_neighbors - self.num_users) if self.config["data"] in ["wikipedia", "reddit"] else src_neighbors
        destinations_batch = torch.where(destinations_batch == 0, 0, destinations_batch - self.num_users) if self.config["data"] in ["wikipedia", "reddit"] else destinations_batch
        
        
        user_emb = self.dyn_user_emb[sources_batch]
        if not self.eth_flg:
            item_emb = self.dyn_item_emb[destinations_batch]
            prev_emb = self.dyn_item_emb[src_neighbors]
        else:
            item_emb = self.dyn_user_emb[destinations_batch]
            prev_emb = self.dyn_user_emb[src_neighbors]
        
        
        # TODO think about window normalizations
        item_prev_sum = self.model(
                                prev_emb,
                                mode='prev',
                                freq=torch.ones_like(src_neighbors).type(torch.FloatTensor),
                                item_max=self.config["item_max"],
                                item_pow=self.config["item_pow"])
        
        item_stat = self.model(mode='stat',
                            freq=torch.ones_like(src_neighbors).type(torch.FloatTensor),
                            item_stat=self.stat_item_emb[src_neighbors] if not self.eth_flg else self.stat_user_emb[src_neighbors],
                            item_max=self.config["item_max"],
                            item_pow=self.config["item_pow"])
        
        
        item_pred_emb = self.model(
                                torch.cat([user_emb, item_prev_sum], dim=1).detach(),
                                mode='pred',
                                item_stat=item_stat,
                                user_stat=self.stat_user_emb[sources_batch],
                                freq=torch.ones(item_prev_sum.size(0)),
                                user_max=self.config["user_max"],
                                user_pow=self.config["user_pow"]
                            )
        
        pos_probs = None
        neg_probs = None
        if negatives is not None:
            if not self.eth_flg:
                pos_dyn = self.dyn_item_emb[destinations_batch]
                pos_stat = self.stat_item_emb[destinations_batch]
            else:
                pos_dyn = self.dyn_user_emb[destinations_batch]
                pos_stat = self.stat_user_emb[destinations_batch]
            
            negatives = torch.Tensor(negatives).type(torch.LongTensor)
            negatives = torch.where(negatives == 0, 0, negatives - self.num_users) if self.config["data"] in ["wikipedia", "reddit"] else negatives
            if not self.eth_flg:
                neg_dyn = self.dyn_item_emb[negatives]
                neg_stat = self.stat_item_emb[negatives]
            else:
                neg_dyn = self.dyn_user_emb[negatives]
                neg_stat = self.stat_user_emb[negatives]
            
            pos_probs = ((torch.cat([pos_dyn, pos_stat], dim=1) - item_pred_emb)**2).sum(1).detach()
            neg_probs = ((torch.cat([neg_dyn, neg_stat], dim=1) - item_pred_emb)**2).sum(1).detach()
        
        loss += ((item_pred_emb - torch.cat([item_emb,
                   self.stat_item_emb[destinations_batch]] if not self.eth_flg else self.stat_user_emb[destinations_batch],
                  dim = 1
                 ).detach()) ** 2).sum(dim=1).mean()
        
        edge_feats = torch.Tensor(self.edge_features.edge_table[edge_idxs_batch])
        
        user_emb_nxt = self.model(user_emb,
                                 torch.cat([item_emb,
                                            (timestamps_batch - src_edge_times[:, -1]).reshape(-1,1),
                                            edge_feats],
                                           dim=1
                                          ).detach(),
                                 mode='user'
                                )
        item_emb_nxt = self.model(torch.cat([user_emb,
                                            (timestamps_batch - dst_edge_times[:, -1]).reshape(-1,1),
                                            edge_feats],
                                           dim=1
                                          ).detach(),
                                 item_emb,
                                 mode='item'
                                )

        
        item_emb_pre = self.model(item_emb,
                                 prev_emb,
                                 mode='addi',
                                 freq=torch.ones_like(src_neighbors).type(torch.FloatTensor),
                                 item_max=self.config["item_max"],
                                 item_pow=self.config["item_pow"])
        
        loss += ((user_emb_nxt - user_emb) ** 2).sum(dim=1).mean()
        loss += ((item_emb_nxt - item_emb) ** 2).sum(dim=1).mean()
        
        self.dyn_user_emb[sources_batch] = user_emb_nxt
        if not self.eth_batch:
            self.dyn_item_emb[destinations_batch] = item_emb_nxt
            self.dyn_item_emb[src_neighbors] = item_emb_pre
        else:
            self.dyn_user_emb[destinations_batch] = item_emb_nxt
            self.dyn_user_emb[src_neighbors] = item_emb_pre
        
        return loss, pos_probs, neg_probs

    def get_inference_params(self):
        return {'n_neighbors':self.config['n_degree']}

    def get_decoder(self, emb_dim):
        return MLP(emb_dim, drop = self.config['drop_out'])
    
    
    def get_results_path(self,):
        return os.path.join(f'./logs/{self.model_id}.pkl')

    def get_checkpoint_path(self,epoch, part='graph', final=False):
        """
        If epoch<0, the training is done and we want path to final model
        """

        if not os.path.isdir('./saved_checkpoints'):
            os.mkdir('./saved_checkpoints')
        if epoch<0:
            return f'./saved_checkpoints/HiLi-{self.model_id}-{part}-final.pth'    
        else:
            return f'./saved_checkpoints/HiLi-{self.model_id}-{part}-{epoch}.pth' 