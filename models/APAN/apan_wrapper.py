import time
from collections import defaultdict
import math
import torch

from models.APAN.model import *
from models.APAN.utils.utils import get_current_ts
from dgl import backend

from data_tools.data_interface import GraphContainer
from model_wrapper import ModelWrapper
from models.tgn.utils import get_neighbor_finder, EarlyStopMonitor, MLP
import torch.nn.functional as F
import evaluation
import numpy as np
import dgl
import os


class Args(object):
    pass

class APANWrapper(ModelWrapper):
    default_arg_config = {
        "emb_dim": 172,
        "len_mail": 10,
        "no_time": False,
        "no_pos": False,
        "n_head": 2,
        "dropout": 0.1,
        "use_mask": False,
        "uniform": True,
        "lr": 3e-4,
        "patience": 5,
        "n_epoch": 50,
        "n_degree": 10,
        "n_layer": 1
    }
    def __init__(self, config=None, model_id=None):
        self.config = {**self.default_arg_config}
        if not (config is None):
            self.new_config = config
            self.config = {**self.config, **self.new_config}
        if model_id is None:
            self.model_id = f'apan_at_{time.ctime()}'
        else:
            self.model_id = f'{model_id} at {time.ctime()}'
        self.model_id = self.model_id.replace(':', '_')
        self.logger = self.prepare_logger(self.model_id)

    def initialize_model(self, full_data, train_data, node_features, edge_features, batch_params,
                         data_setting_mode='transductive'):
        self.device = f'cuda:{self.config["gpu"]}' if torch.cuda.is_available() and not self.config[
            'force_cpu'] else 'cpu'
        
        self.num_nodes = full_data.n_unique_nodes
        self.num_node_feats = node_features.get().shape[1]
        self.config["emb_dim"] = self.num_node_feats
        self.num_edge_feats = edge_features.get().shape[1]
        
        self.args = Args()
        self.args.no_time = self.config["no_time"]
        self.args.no_pos = self.config["no_pos"]
        self.args.n_mail = self.config["len_mail"]
        self.args.dropout = self.config["dropout"]
        self.args.n_layer = self.config["n_layer"]
        
        self.node_features = torch.Tensor(node_features.get()[1:])
        self.edge_features = torch.Tensor(edge_features.get()[:])
        
        self.model = Encoder(self.args, self.config["emb_dim"], n_head=self.config["n_head"], dropout=self.config["dropout"],
                               use_mask=self.config["use_mask"])
        self.args.tasks = "LP"
        self.lp = MLP(2 * self.config["emb_dim"], drop = self.config['dropout'])
        
        self.full_ngh_finder = get_neighbor_finder(full_data, uniform=self.config['uniform'])
        
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.lp.parameters()), lr=self.config['lr'])
        self.batch_params = batch_params
        self.negative_sampler = dgl.dataloading.negative_sampler.Uniform(1)
        self.msg2mail = Msg2Mail(self.args, self.config["emb_dim"])
        
        self.g = dgl.graph(list(zip(range(self.num_nodes), range(self.num_nodes))))
        self.g.edata["timestamp"] = torch.ones(self.num_nodes)
        self.g.ndata["feat"] = self.node_features
        # torch.zeros((self.num_nodes, self.config["emb_dim"]), dtype=torch.float32) # init as zero, people can init it using others.
        self.g.ndata["mail"] = torch.zeros((self.num_nodes, self.config["len_mail"], self.config["emb_dim"] + 2), dtype=torch.float32) 
        self.g.ndata["last_update"] = torch.zeros((self.num_nodes), dtype=torch.float32) 
        

    def compute_edge_probabilities(self, batch, negatives, eval_mode, data_setting_mode, **model_params):
        (
            sources_batch, 
            destinations_batch, 
            timestamps_batch, 
            edge_idxs_batch, 
            _,
        ) = batch
        
        self.g = self.g.add_self_loop()
        
        self.g.add_edges(sources_batch, destinations_batch, {"timestamp": torch.Tensor(timestamps_batch + 1)})
        pos_eids = self.g.edge_ids(sources_batch, destinations_batch)
        selfs = np.concatenate([sources_batch, destinations_batch, negatives])
        self_eids = self.g.edge_ids(selfs, selfs)
        pos_graph = self.g.edge_subgraph(torch.cat([pos_eids, self_eids]))

        self.g.add_edges(sources_batch, negatives, {"timestamp": torch.Tensor(timestamps_batch + 1)})
        neg_eids = self.g.edge_ids(sources_batch, negatives)
        neg_graph = self.g.edge_subgraph(torch.cat([neg_eids, self_eids]))
        self.g.remove_edges(neg_eids)
        self.g.remove_edges(pos_eids)

        current_ts, pos_ts, num_pos_nodes = get_current_ts(self.args, pos_graph, neg_graph)
        pos_graph.ndata['ts'] = current_ts
        emb, _ = self.model(dgl.add_reverse_edges(pos_graph), dgl.add_reverse_edges(neg_graph), len(destinations_batch))
        tmp = torch.Tensor(self.g.ndata['feat'].detach().numpy())
        tmp[pos_graph.ndata[dgl.NID]] = emb.to('cpu')
        pos = self.lp(torch.cat([tmp[sources_batch], tmp[destinations_batch]], dim=1)).sigmoid()
        neg = self.lp(torch.cat([tmp[sources_batch], tmp[negatives]], dim=1)).sigmoid()
        return pos, neg

    def load_model(self, model_path):
        if not (model_path is None) and os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.lp.load_state_dict(torch.load(model_path.replace("graph", "lp")))
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
        self.logger.info('APAN models loaded')
        self.logger.info('Start training node classification task')

        self.decoder = self.get_decoder(self.config["emb_dim"])
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.config['lr_decoder'])
        self.decoder = self.decoder.to(self.device)
        self.decoder_loss_criterion = torch.nn.BCELoss()
        
        train_metrics = defaultdict(list)
        
        early_stopper = EarlyStopMonitor(max_round=self.config['patience'])
        for epoch in range(self.config['n_epoch']):
            start_epoch = time.time()
            self.model = self.model.eval()
            self.decoder = self.decoder.train()
            loss = 0
            
            self.g = dgl.graph(list(zip(range(self.num_nodes), range(self.num_nodes))))
            self.g.edata["timestamp"] = torch.ones(self.num_nodes)
            self.g.ndata["feat"] = self.node_features
            # torch.zeros((self.num_nodes, self.config["emb_dim"]), dtype=torch.float32) # init as zero, people can init it using others.
            self.g.ndata["mail"] = torch.zeros((self.num_nodes, self.config["len_mail"], self.config["emb_dim"] + 2), dtype=torch.float32) 
            self.g.ndata["last_update"] = torch.zeros((self.num_nodes), dtype=torch.float32) 
            
            for i, batch in enumerate(train_data(**self.batch_params['train'])):
                self.decoder_optimizer.zero_grad()
                self.decoder = self.decoder.train()
                    
                (
                    sources_batch, 
                    destinations_batch, 
                    timestamps_batch, 
                    edge_idxs_batch, 
                    labels,
                ) = batch
                
                
                self.g.add_edges(sources_batch, destinations_batch, {"timestamp": torch.Tensor(timestamps_batch + 1), "feat": self.edge_features[edge_idxs_batch]})
                frontier = self.g.in_subgraph(sources_batch)
                pos_eids = self.g.edge_ids(sources_batch, destinations_batch)
                selfs = np.concatenate([sources_batch, destinations_batch])
                self_eids = self.g.edge_ids(selfs, selfs)
                pos_graph = self.g.edge_subgraph(torch.cat([pos_eids, self_eids]))
                
                current_ts, pos_ts, num_pos_nodes = get_current_ts(self.args, pos_graph, None)
                pos_graph.ndata['ts'] = current_ts
                
                
                emb, _ = self.model(dgl.add_reverse_edges(pos_graph), None, len(destinations_batch))
                logits = self.decoder(emb).sigmoid()
                
                tmp = torch.zeros((self.num_nodes,))
                tmp[pos_graph.ndata[dgl.NID]] = logits
                dec_loss = self.decoder_loss_criterion(tmp[sources_batch], torch.from_numpy(labels).float())
                # dec_loss = self.decoder_loss_criterion(logits, torch.from_numpy(labels).float().reshape(-1, 1))

                dec_loss.backward()
                self.optimizer.step()
                loss += dec_loss.item()
                
                # MSG Passing
                with torch.no_grad():
                    # pos_graph.edata["feat"] = torch.zeros((pos_graph.num_edges(), self.config["emb_dim"]))
                    mail = self.msg2mail.gen_mail(self.args, emb, np.concatenate([sources_batch, destinations_batch]), pos_graph, frontier, 'train')
                    if not self.args.no_time:
                        self.g.ndata['last_update'][pos_graph.ndata[dgl.NID][:num_pos_nodes]] = pos_ts.to('cpu')
                    self.g.ndata['feat'][pos_graph.ndata[dgl.NID]] = emb.to('cpu')
                    self.g.ndata['mail'][np.concatenate([sources_batch, destinations_batch])] = mail
            train_metrics['train_losses'].append(loss / i+1)
            _, _ = self.model.eval(), self.decoder.eval()
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
            
            self.g = dgl.graph(list(zip(range(self.num_nodes), range(self.num_nodes))))
            self.g.edata["timestamp"] = torch.ones(self.num_nodes)
            self.g.ndata["feat"] = self.node_features
            # torch.zeros((self.num_nodes, self.config["emb_dim"]), dtype=torch.float32) # init as zero, people can init it using others.
            self.g.ndata["mail"] = torch.zeros((self.num_nodes, self.config["len_mail"], self.config["emb_dim"] + 2), dtype=torch.float32) 
            self.g.ndata["last_update"] = torch.zeros((self.num_nodes), dtype=torch.float32) 
            
            for i, batch in enumerate(training_data):
                self.optimizer.zero_grad()
                self.model = self.model.train()
                self.lp = self.lp.train()
                if i % 100 == 0:
                    print(f'{i}/{training_data.num_batches} batches passed ... It took {round(time.time() - start if start else 0, 2)} seconds')
                    start = time.time()
                    
                (
                    sources_batch, 
                    destinations_batch, 
                    timestamps_batch, 
                    edge_idxs_batch, 
                    _,
                ) = batch
                
                
                self.g.add_edges(sources_batch, destinations_batch, {"timestamp": torch.Tensor(timestamps_batch + 1), "feat": self.edge_features[edge_idxs_batch]})
                frontier = self.g.in_subgraph(sources_batch)
                pos_eids = self.g.edge_ids(sources_batch, destinations_batch)
                
                size = len(sources_batch)
                _, negatives_batch = train_sampler.sample(size)
                
                selfs = np.concatenate([sources_batch, destinations_batch, negatives_batch])
                self_eids = self.g.edge_ids(selfs, selfs)
                pos_graph = self.g.edge_subgraph(torch.cat([pos_eids, self_eids]))
                
                self.g.add_edges(sources_batch, negatives_batch, {"timestamp": torch.Tensor(timestamps_batch + 1)})
                neg_eids = self.g.edge_ids(sources_batch, negatives_batch)
                neg_graph = self.g.edge_subgraph(torch.cat([neg_eids, self_eids]))
                self.g.remove_edges(neg_eids)
                
                current_ts, pos_ts, num_pos_nodes = get_current_ts(self.args, pos_graph, neg_graph)
                pos_graph.ndata['ts'] = current_ts
                
                emb, _ = self.model(pos_graph, neg_graph, num_pos_nodes)
                
                tmp = torch.zeros(self.num_nodes, emb.shape[1])
                tmp[pos_graph.ndata[dgl.NID]] = emb
                
                pos_probs = self.lp(torch.cat([tmp[sources_batch], tmp[destinations_batch]], dim=1)).sigmoid()
                neg_probs = self.lp(torch.cat([tmp[sources_batch], tmp[negatives_batch]], dim=1)).sigmoid()
                
                with torch.no_grad():
                    pos_label = torch.ones_like(pos_probs, dtype=torch.float, device= self.device)
                    neg_label = torch.zeros_like(neg_probs, dtype=torch.float, device=self.device)
                
                loss = self.criterion(pos_probs, pos_label) + self.criterion(neg_probs, neg_label)

                loss.backward()
                self.optimizer.step()
                m_loss.append(loss.item())
                
                # MSG Passing
                with torch.no_grad():
                    # pos_graph.edata["feat"] = torch.zeros((pos_graph.num_edges(), self.config["emb_dim"]))
                    mail = self.msg2mail.gen_mail(self.args, emb, np.concatenate([sources_batch, destinations_batch]), pos_graph, frontier, 'train')

                    if not self.args.no_time:
                        self.g.ndata['last_update'][pos_graph.ndata[dgl.NID]] = pos_ts.to('cpu')
                    self.g.ndata['feat'][pos_graph.ndata[dgl.NID]] = emb.to('cpu')
                    self.g.ndata['mail'][np.concatenate([sources_batch, destinations_batch])] = mail
                
                
                
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
                best_lp_path = self.get_checkpoint_path(self.early_stopper.best_epoch, 'lp')
                self.model.load_state_dict(torch.load(best_model_path))
                self.lp.load_state_dict(torch.load(best_lp_path))
                torch.save(self.model.state_dict(), self.get_checkpoint_path(-1, 'graph'))
                torch.save(self.lp.state_dict(), self.get_checkpoint_path(-1, 'lp'))
                self.logger.info(f'Loaded the best model at epoch {self.early_stopper.best_epoch} for inference')
                self.model.eval()
                break
            elif (epoch+1==self.config['n_epoch']):
                self.model.eval()
                
            torch.save(self.model.state_dict(), self.get_checkpoint_path(epoch, 'graph'))
            torch.save(self.lp.state_dict(), self.get_checkpoint_path(epoch, 'lp'))
        return train_metrics

    def compute_clf_probability(self, data, eval_mode, data_setting_mode, **model_params):
        (
            sources_batch, 
            destinations_batch, 
            timestamps_batch, 
            edge_idxs_batch, 
            _,
        ) = data
        
        
        self.g.add_edges(sources_batch, destinations_batch, {"timestamp": torch.Tensor(timestamps_batch + 1)})
        pos_eids = self.g.edge_ids(sources_batch, destinations_batch)
        selfs = np.concatenate([sources_batch, destinations_batch])
        self_eids = self.g.edge_ids(selfs, selfs)
        pos_graph = self.g.edge_subgraph(torch.cat([pos_eids, self_eids]))
        self.g.remove_edges(pos_eids)

        current_ts, pos_ts, num_pos_nodes = get_current_ts(self.args, pos_graph, None)
        pos_graph.ndata['ts'] = current_ts
        emb, _ = self.model(dgl.add_reverse_edges(pos_graph), None, len(destinations_batch))
        tmp = torch.Tensor(self.g.ndata['feat'].detach().numpy())
        tmp[pos_graph.ndata[dgl.NID]] = emb.to('cpu')
        return self.decoder(tmp[sources_batch]).sigmoid()

    def get_inference_params(self):
        return {'n_neighbors': self.config['n_degree']}

    def get_decoder(self, emb_dim):
        self.args.tasks = "NC"
        return MLP(emb_dim, drop = self.config['dropout'])

    def get_results_path(self, ):
        return os.path.join(f'./logs/{self.model_id}.pkl')

    def get_checkpoint_path(self, epoch, part='graph', final=False):
        """
        If epoch<0, the training is done and we want path to final model
        """

        if not os.path.isdir('./saved_checkpoints'):
            os.mkdir('./saved_checkpoints')
        if epoch < 0:
            return f'./saved_checkpoints/APAN-{self.model_id}-{part}-final.pth'
        else:
            return f'./saved_checkpoints/APAN-{self.model_id}-{part}-{epoch}.pth'
