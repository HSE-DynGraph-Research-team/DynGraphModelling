

from data_tools.sampling import get_samplers
import evaluation
from data_tools.data_processing import prepare_transductive_data_from_transact, prepare_inductive_data_from_transact
import pickle as pkl
import torch
import gc
from copy import deepcopy
import numpy as np




def report_metrics(metrics):
    #TODO make better reporting
    for k,v in metrics.items():
        print(f'{k}: {v}')

def prepare_data_for_training(
    split_params=None, 
    node_mask_frac=0.1,
    edge_mask_frac=0.1,
    data_params={},
    data_setting_mode='transductive'
    ):

    if data_setting_mode=='transductive':
        prepare_func = prepare_transductive_data_from_transact
    elif data_setting_mode=='inductive':
        prepare_func = prepare_inductive_data_from_transact
    data = prepare_func(
        split_params=split_params, 
        node_mask_frac=node_mask_frac, 
        edge_mask_frac=edge_mask_frac, 
        data_params=data_params)
    
    (
        node_features, 
        edge_features, 
        full_data, 
        train_data, 
        val_data, 
        test_data
        ) = data

    samplers = get_samplers(*data[2:], data_setting_mode=data_setting_mode)
    (
        train_sampler, 
        val_sampler, 
        test_sampler, 
        ) = samplers

    return node_features, edge_features, full_data, train_data, val_data, test_data, train_sampler, val_sampler, test_sampler



def single_mode_pipeline(
    model_wrapper, 
    batch_params=None, 
    split_params=None, 
    node_mask_frac=0.1,
    edge_mask_frac=0.1,
    data_params={},
    data_setting_mode='transductive'
    ):
    model_wrapper_ = deepcopy(model_wrapper)

    (
        node_features, 
        edge_features, 
        full_data, 
        train_data, 
        val_data, 
        test_data, 
        train_sampler, 
        val_sampler, 
        test_sampler
    ) = prepare_data_for_training(
        split_params=split_params,
        node_mask_frac=node_mask_frac,
        edge_mask_frac=edge_mask_frac,
        data_params=data_params,
        data_setting_mode=data_setting_mode,
    )
    if len(np.unique(full_data.label_table[:,2]))==1:
        do_clf=False
    else:
        do_clf=True
    model_wrapper_.initialize_model(full_data, train_data, node_features, edge_features, batch_params, data_setting_mode=data_setting_mode)
    is_trained = model_wrapper_.load_model(data_params.get('model_path', None))
    train_metrics = model_wrapper_.train_model(train_data, val_data, train_sampler, val_sampler, do_clf)

    inference_params = model_wrapper_.get_inference_params()

    metrics_edge = evaluation.eval_edge_prediction(        model_wrapper_, test_data, test_sampler,  data_setting_mode=data_setting_mode, batch_params=batch_params['test'], eval_mode='test', **inference_params)
    if do_clf:
        metrics_clf  = evaluation.eval_node_bin_classification(model_wrapper_, test_data,                data_setting_mode=data_setting_mode, batch_params=batch_params['test'], eval_mode='test', **inference_params)
    else:
        metrics_clf={}
    return train_metrics, metrics_edge, metrics_clf



def combined_pipeline(
    model_wrapper, 
    data_path='./data/', 
    model_path=None, 
    batch_params=None, 
    split_params=None, 
    node_mask_frac=0.1,
    edge_mask_frac=0.1,
    data_params={},
    t_i_setting='ti',
    ):

    if t_i_setting is None:
        run_trans, run_ind = True,True
    else:
        run_trans = True if 't' in t_i_setting else False
        run_ind = True if 'i' in t_i_setting else False

    if batch_params is None:
        raise ValueError('Specify batch params')
    data_params['data_path'] = data_path
    data_params['model_path'] = model_path


    train_metrics = {}
    if run_trans:
        t_train_metrics, t_metrics_edge, t_metrics_clf = single_mode_pipeline(
            model_wrapper, 
            batch_params=batch_params,
            split_params=split_params,
            node_mask_frac=node_mask_frac,
            edge_mask_frac=edge_mask_frac,
            data_params=data_params,
            data_setting_mode='transductive'
        )

        trans_metrics = {
            **{'transductive_'+x:y for x,y in t_train_metrics.items()},
            **{f'trans node_clf {k}':v for k,v in t_metrics_clf.items()},
            **{f'trans edge_pred {k}':v for k,v in t_metrics_edge.items()},
            }
    else:
        trans_metrics={}

    if run_ind:
        i_train_metrics, i_metrics_edge, i_metrics_clf = single_mode_pipeline(
            model_wrapper, 
            batch_params=batch_params,
            split_params=split_params,
            node_mask_frac=node_mask_frac,
            edge_mask_frac=edge_mask_frac,
            data_params=data_params,
            data_setting_mode='inductive'
        )
        ind_metrics = {
            **{'inductive_'+x:y for x,y in i_train_metrics.items()},
            **{f'induc node_clf {k}':v for k,v in i_metrics_clf.items()},
            **{f'induc edge_pred {k}':v for k,v in i_metrics_edge.items()},
            }
    else:
        ind_metrics={}

    results = {
        **trans_metrics,
        **ind_metrics
    }
    if run_trans:
        report_metrics({'Transductive node classification ' + k:v for k,v in t_metrics_clf.items()})
        report_metrics({'Transductive edge prediction '     + k:v for k,v in t_metrics_edge.items()})
    if run_ind:
        report_metrics({'Inductive node classification '    + k:v for k,v in i_metrics_clf.items()})
        report_metrics({'Inductive edge prediction '        + k:v for k,v in i_metrics_edge.items()})

    results['model_id'] = model_wrapper.model_id
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    with open(f'./logs/{model_wrapper.model_id}.pkl', 'wb') as f:
        pkl.dump(results, f)
    return results



if __name__ == '__main__':
    from models.tgn_new.tgn_wrapper import TGNCAWWrapper
    # from models.tgn.tgn_wrapper import TGNWrapper
    # from models.CAW.caw_wrapper import CAWWrapper
    # from models.tdgnn.tdgnn_wrapper import TDGNNWrapper

    data_name='wikipedia'
    model_wrap=TGNCAWWrapper(config={
        'force_cpu':False, 
        'n_epoch':30,
        'n_layer':1,
        'use_caw':True,
        'caw_layers': 1,
        'caw_neighbors':['8', '2'],
        'use_caw_lstm': True,
})
        # 'data_mode':'transaction', 
        # 'init_params':{
        #     'config':{



    # model_wrap = CAWWrapper( model_id=data_name, config={'n_epoch':1, 'force_cpu':False, 'n_layer':1})
    default_batch_params = {
        'train':{
            'batch_size':32,
            'batch_spec':None,
            'divide_by':'edge_ix',
            'data_mode': 'transaction'
        },
        'val':{
            'batch_size':32,
            'batch_spec':None,
            'divide_by':'edge_ix',
            'data_mode': 'transaction'
        },
        'test':{
            'batch_size':32,
            'batch_spec':None,
            'divide_by':'edge_ix',
            'data_mode': 'transaction'
        }
    }
    default_data_params = {
        'randomize_features':False,
        'fetcher':'transact',
        'dataset_name':'wikipedia',
    }

    # model_path = './saved_checkpoints/tgn_at_Tue Mar 30 18:08:59 2021-graph-wikipedia-1.pth'
    # self_supervised_pipeline(tgn_wrap)
    res = combined_pipeline(model_wrap, batch_params=default_batch_params, data_params=default_data_params)
