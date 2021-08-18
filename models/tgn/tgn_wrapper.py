from model_wrapper import ModelWrapper
from data_tools.data_processing import compute_time_statistics
from .utils import get_neighbor_finder, EarlyStopMonitor, MLP
import evaluation
import torch
import time
import os
import math
import numpy as np
from .model.tgn import TGN
from collections import defaultdict






class TGNWrapper(ModelWrapper):

    default_arg_config = {
        'data_mode':'transaction',
        'data':'wikipedia',
        'bs':200,
        'prefix':'',
        'n_degree':10,
        'n_head':2,
        'n_epoch':2,
        'n_layer':1,
        'lr':0.0003,
        'lr_decoder':0.0005,
        'patience':5,
        'n_runs':1,
        'drop_out':0.1,
        'gpu':0,
        'node_dim':100,
        'time_dim':100,
        'backprop_every':1,
        'embedding_module':'graph_attention',#["graph_attention", "graph_sum", "identity", "time"]
        'message_function':'identity',#["mlp", "identity"]
        'memory_updater':'gru',
        'aggregator':'last',
        'message_dim':100,
        'memory_dim':172,
        'use_memory':True,
        'memory_update_at_end':False,
        'different_new_nodes':False,
        'uniform':False,
        'randomize_features':True,
        'use_destination_embedding_in_message':False,
        'use_source_embedding_in_message':False,
        'dyrep':False,
        'use_validation':True,
        'force_cpu':False

        }

    def __init__(self, config=None, model_id=None):
        self.config = {**self.default_arg_config}
        if not (config is None):
            self.new_config=config
            self.config = {**self.config, **self.new_config}
        if model_id is None:
            self.model_id = f'tgn_at_{time.ctime()}'
        else:
            self.model_id = f'{model_id} at {time.ctime()}'
        self.model_id = self.model_id.replace(':', '_')
        self.logger = self.prepare_logger(self.model_id)

    def initialize_model(self, full_data, train_data,node_features, edge_features, batch_params, data_setting_mode):

        (
            mean_time_shift_src, 
            std_time_shift_src, 
            mean_time_shift_dst, 
            std_time_shift_dst 
        ) = compute_time_statistics(
            full_data.edge_table[:,2], 
            full_data.edge_table[:,3], 
            full_data.edge_table[:,1]
        )
        self.device = f'cuda:{self.config["gpu"]}' if torch.cuda.is_available() and not self.config['force_cpu'] else 'cpu'
        self.max_idx = max(full_data.unique_nodes)

        self.data_setting_mode = data_setting_mode

        self.train_ngh_finder = get_neighbor_finder(train_data, uniform=self.config['uniform'], max_node_idx=self.max_idx)
        self.full_ngh_finder  = get_neighbor_finder(full_data, uniform=self.config['uniform'])
        
        self.model = TGN(
            neighbor_finder=self.train_ngh_finder, 
            node_features=node_features,
            edge_features=edge_features,
            device=self.device,
            n_layers=self.config['n_layer'],
            n_heads=self.config['n_head'],
            dropout=self.config['drop_out'],
            use_memory=self.config['use_memory'],
            message_dimension=self.config['message_dim'],
            memory_dimension=self.config['memory_dim'],
            n_neighbors=self.config['n_degree'],
            memory_update_at_start=not self.config['memory_update_at_end'],
            embedding_module_type=self.config['embedding_module'],
            message_function=self.config['message_function'],
            aggregator_type=self.config['aggregator'],
            memory_updater_type=self.config['memory_updater'],
            use_destination_embedding_in_message=self.config['use_destination_embedding_in_message'],
            use_source_embedding_in_message=self.config['use_source_embedding_in_message'],
            dyrep=self.config['dyrep'],
            mean_time_shift_src=mean_time_shift_src,
            std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst,
            std_time_shift_dst=std_time_shift_dst,
        )


        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
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

        if eval_mode!='train':
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
        if eval_mode!='train':
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
        if eval_mode!='train':
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
    
        train_metrics_1 = self.train_self_supervised(train_data, val_data, train_sampler, val_sampler  )
        if do_clf:
            train_metrics_2 = self.train_supervised(train_data, val_data, train_sampler, val_sampler)
        else:
            train_metrics_2 = {}
        return {'train_unsupervised':train_metrics_1, 'train_supervised':train_metrics_2}

    def train_self_supervised(self, train_data, val_data, train_sampler, val_sampler  ):
        num_instance = len(train_data.edge_table)
        num_batch = math.ceil(num_instance / self.config['bs'])

        self.logger.info('num of training instances: {}'.format(num_instance))
        self.logger.info('num of batches per epoch: {}'.format(num_batch))
        idx_list = np.arange(num_instance)

        train_metrics = defaultdict(list)

        self.early_stopper = EarlyStopMonitor(max_round=self.config['patience'])


        keep_training = True
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
            training_data = train_data(**self.batch_params['train'])
            for i, batch in enumerate(training_data):
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
                size = len(sources_batch)
                _, negatives_batch = train_sampler.sample(size)
                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device= self.device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=self.device)



                self.model = self.model.train()
                pos_prob, neg_prob = self.compute_edge_probabilities(
                    batch,
                    negatives_batch,
                    eval_mode='train',
                    data_setting_mode=self.data_setting_mode,
                    n_neighbors = self.config['n_degree']
                    )

                loss += self.criterion(
                    pos_prob.reshape(pos_label.shape), pos_label
                ) + self.criterion(
                    neg_prob.reshape(neg_label.shape), neg_label
                    )



                if (not i%self.config['backprop_every'])&(i):
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

    def train_supervised(self, train_data, val_data, train_sampler, val_sampler):
        num_instance = train_data.edge_table.shape[0]
        num_batch = math.ceil(num_instance / self.config['bs'])

        self.logger.debug('Num of training instances: {}'.format(num_instance))

        self.model.eval()
        self.logger.info('TGN models loaded')
        self.logger.info('Start training node classification task')

        self.decoder = self.get_decoder(self.model.node_raw_features)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.config['lr_decoder'])
        self.decoder = self.decoder.to(self.device)
        self.decoder_loss_criterion = torch.nn.BCELoss()

        train_metrics=defaultdict(list)

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
                                                                                                self.config['n_degree'])

                labels_batch_torch = torch.from_numpy(
                    labels_batch).float().to(self.device)
                pred = self.decoder(source_embedding).sigmoid()
                decoder_loss = self.decoder_loss_criterion(pred, labels_batch_torch)
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

    def get_decoder(self, node_features):
        return MLP(node_features.shape[1], drop = self.config['drop_out'])

    def get_inference_params(self):
        return {'n_neighbors':self.config['n_degree']}

    def get_results_path(self,):
        return os.path.join(f'./logs/{self.model_id}.pkl')

    def get_checkpoint_path(self,epoch, part='graph', final=False):
        """
        If epoch<0, the training is done and we want path to final model
        """

        if not os.path.isdir('./saved_checkpoints'):
            os.mkdir('./saved_checkpoints')
        if epoch<0:
            return f'./saved_checkpoints/TGN-{self.model_id}-{part}-final.pth'    
        else:
            return f'./saved_checkpoints/TGN-{self.model_id}-{part}-{epoch}.pth'    

