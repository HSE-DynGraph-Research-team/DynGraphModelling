from model_wrapper import ModelWrapper
import time
from log_utils import get_logger
from collections import defaultdict
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
import os

from models.CAW.graph import *
from models.CAW.module import CAWN
from models.CAW.utils import EarlyStopMonitor
import torch

class CAWWrapper(ModelWrapper):
    
    default_arg_config = {
        'data_mode':'transaction',
        'agg':'walk',
        'attn_agg_method':'attn',
        'attn_mode':'prod',
        'attn_n_head':2,
        'bias':1e-5,#0.0,
        'bs':64,
        'cpu_cores':1,
        'data':'wikipedia',
        'drop_out':0.1,
        'gpu':0,
        'lr':0.0001,
        'mode':'t',
        'n_degree':['32'],#['64','1'],
        'n_epoch':50,
        'n_layer':2,
        'ngh_cache':False,
        'pos_dim':108,
        'pos_enc':'lp',
        'pos_sample':'binary',
        'seed':0,
        'time':'time',
        'tolerance':0.001,
        'verbosity':1,
        'walk_linear_out':False,
        'walk_mutual':False,
        'walk_n_head':8,
        'walk_pool':'sum',#'attn',
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
        self.logger = self.prepare_logger(self.model_id)
        self.config['walk_mutual'] = True if self.config['walk_pool']=='attn' else False


    def initialize_model(self, full_data, train_data,node_features, edge_features, batch_params, data_setting_mode='transductive'):
        """
        Some model (TGN included) require data to initialize
        Initialize model there; or, if model can be init'd without data, do it in the __init__

        full_data: GraphContainer object, containing full dataset
        train_data: GraphContainer object, containing train dataset
        node_features: NodeFeatures object, used for lookup on node features by id & timestamp (see data_interface.py)
        edge_features: EdgeFeatures object, used for lookup on edge features by id & timestamp
        batch_params: Dict, batching config for train/val/test
        mode: str, {'transductive','inductive'} - flag to indicate the current mode; may be used inside wrapper
        """

        self.device = f'cuda:{self.config["gpu"]}' if torch.cuda.is_available() and not self.config['force_cpu'] else 'cpu'
        self.max_idx = max(full_data.unique_nodes)
        self.data_setting_mode = data_setting_mode
        self.full_ngh_finder = get_neighbor_finder(full_data, max_node_idx=None,bias=self.config['bias'], use_cache=self.config['ngh_cache'], sample_method=self.config['pos_sample'])
        self.batch_params = batch_params


        self.model = CAWN(
            node_features, 
            edge_features, 
            agg=self.config['agg'],
            num_layers=self.config['n_layer'], 
            use_time=self.config['time'], 
            attn_agg_method=self.config['attn_agg_method'], 
            attn_mode=self.config['attn_mode'],
            n_head=self.config['attn_n_head'],
            drop_out=self.config['drop_out'], 
            pos_dim=self.config['pos_dim'], 
            pos_enc=self.config['pos_enc'],
            num_neighbors=self.config['n_degree'], 
            walk_n_head=self.config['walk_n_head'], 
            walk_mutual=self.config['walk_mutual'], 
            walk_linear_out=self.config['walk_linear_out'],
            cpu_cores=self.config['cpu_cores'], 
            verbosity=self.config['verbosity'], 
            )
        self.model.to(self.device)

        return None


    def compute_edge_probabilities(
            self,
            batch,
            negatives,
            eval_mode,
            data_setting_mode,
            **model_params):
        """
        Predict 2 batches of edges - from source to destination and from source to negative samples
        We separate real and sample edges, as some models (e.g. TGN) need to know, which data is real to utilize it afterwards
        batch: tuple of transaction data (source_id, dest_id, timestamps, edge_idx) or PyTorchGeometric graph snapshot
        eval_mode: str, {'train','val','test'} to signal whether we are evaluating (e.g. put model.eval() for pytorch models if not train)
        data_setting_mode: str, {'transductive','inductive'}, indicates which setting is used
        negatives: batch of negative samples edges (id of destination node) of the same edge as the batch
        **model_params: any parameters the model needs for evaluation
        """

        (
            sources_batch, 
            destinations_batch, 
            timestamps_batch, 
            edge_idxs_batch, 
            _,
            ) = batch


        real_probabilities, sampled_probabilities = self.model.contrast(
            sources_batch, 
            destinations_batch, 
            negatives, 
            timestamps_batch, 
            edge_idxs_batch,
            eval_mode=='test')
        return real_probabilities, sampled_probabilities


    def load_model(self, model_path):
        """
        Load model weights and return True, or finish initialization of model and return False
        """
        is_trained = False
        return is_trained


    def train_model(self, train_data, val_data, train_sampler, val_sampler, do_clf):
        """
        All the training happens here
        train_data: GraphContainer object, suitable for training
        val_data: GraphContainer object, suitable for validation
        new_node_val_data: GraphContainer object, suitable for validation on new nodes
        train_sampler: sampler of random edges to use during training
        val_sampler: sampler of random edges to use during validation

        returns dict of training metrics
        """

        self.partial_ngh_finder = get_neighbor_finder([train_data, val_data], max_node_idx=None,bias=self.config['bias'], use_cache=self.config['ngh_cache'], sample_method=self.config['pos_sample'])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.criterion = torch.nn.BCELoss()
        self.early_stopper = EarlyStopMonitor(tolerance=self.config['tolerance'])


        train_metrics = defaultdict(list)

        if self.data_setting_mode == 'transductive':  # transductive
            self.model.update_ngh_finder(self.full_ngh_finder)
        elif self.data_setting_mode == 'inductive':  # inductive
            self.model.update_ngh_finder(self.partial_ngh_finder)
        else:
            raise ValueError('training mode {} not found.'.format(mode))

        for epoch in range(self.config['n_epoch']):
            start_epoch = time.time()
            pred_scores = []
            pred_labels = []
            true_labels = []
            losses = []
            # np.random.shuffle(idx_list)  # shuffle the training samples for every epoch
            self.logger.info(f'start {epoch} epoch')
            for i, batch in enumerate(train_data(**self.batch_params['train'])):
                times = []
                size = len(batch[0])
                time_since_epoch_start = time.time()-start_epoch
                mean_time = (time_since_epoch_start)/(i+1)
                batch_perc = size*(i+1)/train_data.edge_table.shape[0]
                batch_num = (train_data.edge_table.shape[0]/size)*(i+1)
                time_left = mean_time * (1-batch_perc) * (train_data.edge_table.shape[0]/size)
                if i%100==0:
                    self.logger.info(f'start {i} batch({round(batch_perc*100,3)}%, {round(time_since_epoch_start/60,2)} mins since start, mean time {round(mean_time,2)}, expected to finish in {round(time_left/60, 2)} mins)')
                _, negatives = train_sampler.sample(size)
                
                # feed in the data and learn from error
                self.optimizer.zero_grad()
                self.model.train()
                # times.append(time.time())
                pos_prob, neg_prob = self.compute_edge_probabilities(
                    batch,
                    negatives,
                    data_setting_mode=self.data_setting_mode,
                    eval_mode='train'
                )
                # times.append(time.time())
                # self.model.contrast(
                #     sources_batch, 
                #     destinations_batch, 
                #     neg_destinations_batch, 
                #     timestamps_batch, 
                #     edge_idxs_batch)   # the core training code
                pos_label = torch.ones(size, dtype=torch.float, device=self.device, requires_grad=False)
                neg_label = torch.zeros(size, dtype=torch.float, device=self.device, requires_grad=False)
                # times.append(time.time())
                loss = self.criterion(pos_prob, pos_label) + self.criterion(neg_prob, neg_label)
                # times.append(time.time())
                loss.backward()
                # times.append(time.time())
                self.optimizer.step()

                # collect training results
                # times.append(time.time())

                with torch.no_grad():
                    pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                    pred_label = pred_score > 0.5
                    true_label = np.concatenate([np.ones(size), np.zeros(size)])
                    pred_scores.extend(pred_score)
                    pred_labels.extend(pred_label)
                    true_labels.extend(true_label)
                    losses.append(loss.item())
                # times.append(time.time())
                # self.logger.info([times[i+1]-times[i] for i in range(len(times)-1)])
            epoch_time = time.time() - start_epoch
            train_metrics['epoch_times'].append(epoch_time)

            pred_score = np.array(pred_scores)
            pred_label = np.array(pred_labels)
            true_label = np.array(true_labels)
            losses = np.array(losses)
            
            train_acc, train_ap, train_f1, train_auc, train_pr, train_rec = self.make_metrics(true_label, pred_label, pred_score)

            train_metrics['train_f1'].append(train_f1)
            train_metrics['train_pr'].append(train_pr)
            train_metrics['train_rec'].append(train_rec)
            train_metrics['train_acc'].append(train_acc)
            train_metrics['train_auc'].append(train_auc)
            train_metrics['train_ap'].append(train_ap)

            train_metrics['train_losses'].append(losses.mean())


            # validation phase use all information
            self.model.eval()
            val_pred=[]
            val_true=[]
            val_score=[]
            with torch.no_grad():
                for i, batch in enumerate(val_data(**self.batch_params['val'])):
                    size = len(batch[0])
                    _, negatives = val_sampler.sample(size)

                    # feed in the data and learn from error
                    pos_prob, neg_prob = self.compute_edge_probabilities(
                        batch,
                        negatives,
                        data_setting_mode=self.data_setting_mode,
                        eval_mode='val'
                    )
                    pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                    pred_label = pred_score > 0.5
                    true_label = np.concatenate([np.ones(size), np.zeros(size)])
                    val_pred.extend(pred_label)
                    val_true.extend(true_label)
                    val_score.extend(pred_score)
                val_pred = np.array(val_pred)
                val_true = np.array(val_true)
                val_score = np.array(val_score)
                val_acc, val_ap, val_f1, val_auc, val_pr, val_rec = self.make_metrics(val_true, val_pred, val_score)
            
            self.logger.info('epoch: {}:'.format(epoch))
            self.logger.info('epoch mean loss: {}'.format(np.mean(losses)))
            self.logger.info('train acc: {}, val acc: {}'.format(train_acc, val_acc))
            self.logger.info('train auc: {}, val auc: {}'.format(train_auc, val_auc))
            self.logger.info('train ap: {}, val ap: {}'.format(train_ap, val_ap))
            self.logger.info('val f1: {}'.format(val_f1))
            self.logger.info('val pr: {}'.format(val_pr))
            self.logger.info('val rec: {}'.format(val_rec))

            train_metrics['val_f1'].append(val_f1)
            train_metrics['val_pr'].append(val_pr)
            train_metrics['val_rec'].append(val_rec)
            train_metrics['val_acc'].append(val_acc)
            train_metrics['val_auc'].append(val_auc)
            train_metrics['val_ap'].append(val_ap)




            if epoch == 0:
                # save things for data anaysis
                checkpoint_dir = '/'.join(self.get_checkpoint_path(0).split('/')[:-1])
                self.model.ngh_finder.save_ngh_stats(checkpoint_dir)  # for data analysis
                self.model.save_common_node_percentages(checkpoint_dir)

            # early stop check and checkpoint saving
            torch.save(self.model.state_dict(), self.get_checkpoint_path(epoch,'main' ))
            if  (epoch+1==self.config['n_epoch']) or self.early_stopper.early_stop_check(val_ap):
                if self.early_stopper.early_stop_check(val_ap):
                    self.logger.info(f'No improvment over {self.early_stopper.max_round} epochs, stop training')
                self.logger.info(f'Loading the best model at epoch {self.early_stopper.best_epoch}')
                best_checkpoint_path = self.get_checkpoint_path(self.early_stopper.best_epoch, 'main')
                self.model.load_state_dict(torch.load(best_checkpoint_path))
                torch.save(self.model.state_dict(), self.get_checkpoint_path(-1, 'main'))
                self.logger.info(f'Loaded the best model at epoch {self.early_stopper.best_epoch} for inference')
                break


        self.model.eval()
        self.model.update_ngh_finder(self.full_ngh_finder)
        return {'train':train_metrics}


    def make_metrics(self, val_true, val_pred, val_score):

        val_acc = (val_pred == val_true).mean()
        val_ap = average_precision_score(val_true, val_score)
        val_f1 = f1_score(val_true, val_pred)
        val_auc = roc_auc_score(val_true, val_score)
        val_pr = precision_score(val_true, val_pred, zero_division=1)
        val_rec = recall_score(val_true, val_pred)
        return val_acc, val_ap, val_f1, val_auc, val_pr, val_rec



    def get_checkpoint_path(self,epoch, part='graph', final=False):
        """
        If epoch<0, the training is done and we want path to final model
        """

        if not os.path.isdir('./saved_checkpoints'):
            os.mkdir('./saved_checkpoints')
        if epoch<0:
            return f'./saved_checkpoints/CAW-{self.model_id}-{part}-final.pth'    
        else:
            return f'./saved_checkpoints/CAW-{self.model_id}-{part}-{epoch}.pth'    






    def get_inference_params(self):
        """
        If your model requires additional parameters during inference - put them there in a dict
        """
        return dict()


    def compute_clf_probability(
        self, 
        data,
        eval_mode,
        data_setting_mode,
        **model_params
        ):
        """
        Predict probability of binary target of the destination nodes
        data: tuple of transaction data (source_id, dest_id, timestamps, edge_idx) or PyTorchGeometric graph snapshot
        eval_mode: bool to signal whether we are evaluating (e.g. put model.eval() for pytorch models if True)
        data_setting_mode: str, {'transductive','inductive'}, indicates which setting is used
        **model_params: any parameters the model needs for evaluation
        """
        if not hasattr(self, 'clf_warning_given'):
            self.logger.warning('No clf implemented!')
            self.clf_warning_given=True
        
        pred_prob = np.zeros(data[0].shape[0])
        return pred_prob













