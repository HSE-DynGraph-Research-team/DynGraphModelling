
from abc import ABC, abstractmethod
import time
from log_utils import get_logger

class ModelWrapper(ABC):
    default_arg_config = {
        'data_mode':'transaction',
        }


    def __init__(self):
        pass

    @abstractmethod
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
        self.model = None
        return None

    @abstractmethod
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
        real_probabilities, sampled_probabilities = None, None
        return real_probabilities, sampled_probabilities

    @abstractmethod
    def load_model(self, model_path):
        """
        Load model weights and return True, or finish initialization of model and return False
        """
        is_trained = False
        return is_trained

    @abstractmethod
    def train_model(self, train_data, val_data, train_sampler, val_sampler):
        """
        All the training happens here
        train_data: GraphContainer object, suitable for training
        val_data: GraphContainer object, suitable for validation
        new_node_val_data: GraphContainer object, suitable for validation on new nodes
        train_sampler: sampler of random edges to use during training
        train_sampler: sampler of random edges to use during validation

        returns dict of training metrics
        """
        train_metrics = { # put metrics in the following format
            'phase_1':{ # name the phase in an informative way - e.g. unsupervised_train/supervised_train for TGN
                'epoch_times':list(), # list of epoch durations
                'train_losses':list(), # list of epoch losses
                'metric_1':list(), # list of epoch metrics on val
                'metric_2':list(), # different metric
            }
        }
        return train_metrics

    @abstractmethod
    def get_inference_params(self):
        """
        If your model requires additional parameters during inference - put them there in a dict
        """
        return dict()

    @abstractmethod
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
        eval_mode: str, {'train','val','test'} to signal whether we are evaluating (e.g. put model.eval() for pytorch models if not train)
        data_setting_mode: str, {'transductive','inductive'}, indicates which setting is used
        **model_params: any parameters the model needs for evaluation
        """
        pred_prob = None
        return pred_prob



    def prepare_logger(self,model_id):
        return get_logger(model_id)


