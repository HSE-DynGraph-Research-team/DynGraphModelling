import time
import pickle as pkl
import traceback
import os
import sys
import itertools
from copy import deepcopy
from pipelines import combined_pipeline
import log_utils
from models.tgn.tgn_wrapper import TGNWrapper
from models.tgn_new.tgn_wrapper import TGNCAWWrapper
from models.tdgnn.tdgnn_wrapper import TDGNNWrapper
from models.CAW.caw_wrapper import CAWWrapper
from models.TI_GNN.ti_gnn_wrapper import TiGNNWrapper


from itertools import product
def dict_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


def run_scenario(model, dataset_name, batch_params, split_params, node_mask_frac, edge_mask_frac, t_i_setting=None, **kwargs):

    batch_params_ = deepcopy(batch_params)
    data_params = batch_params_.pop('data_params')
    data_params['dataset_name']=dataset_name

    results = combined_pipeline(
        model_wrapper=model, 
        batch_params=batch_params_,
        split_params=split_params,
        node_mask_frac=node_mask_frac,
        edge_mask_frac=edge_mask_frac,
        data_params=data_params, 
        t_i_setting=t_i_setting
        )
    return results


seconds_in_month = int(60*60*24*365/12)
seconds_in_day = int(60*60*24)
configs_dir = './batch_configs_eth'



"""
First number to indicate train-val bound, second - for val-set bound.
E.g. [0.5, 0.8] -> train - 0.5 of data, val - 0.3 of data, test - 0.2 of data
"""


node_mask_settings = {
    # "1%_label_mask" :0.01,
    # "5%_label_mask" :0.05,
    '10%_label_mask':0.1,
    # '25%_label_mask':0.25,
    # '50%_label_mask':0.5,
    '75%_label_mask':0.75,
    # '95%_label_mask':0.95,
    
}
edge_mask_settings = {
    # "1%_edge_mask" :0.01,
    # "5%_edge_mask" :0.05,
    '10%_edge_mask':0.1,
    # '25%_edge_mask':0.25,
    # '50%_edge_mask':0.5,
    '75%_edge_mask':0.75,
    # '95%_edge_mask':0.95,
}

split_param_settings = {
    "split 70-10-20":[0.7,0.8],
}




"""
NB: Yelp dataset does not conform to suitable format yet, so it is excluded from active datasets
"""
batch_param_settings = {
    # 'yelp 50%+monthly':{
    #     'train':{
    #         'batch_size':None,
    #         'batch_spec':[0.5, *[seconds_in_month for _ in range(1000)] ],
    #         'divide_by':'time_window',
    #     },
    #     'val':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_month for _ in range(1000)],
    #         'divide_by':'time_window',
    #     },
    #     'test':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_month for _ in range(1000)],
    #         'divide_by':'time_window',
    #     },
    #    'description': 'Yelp dataset; training dataset is batched in the following fashion: first batch contains half of the whole train sample, the rest are batched monthly. Val and test are batched monthly.',
    #    'big_batch_included':True,
    # },
    # 'yelp monthly':{
    #     'train':{
    #         'batch_size':None,
    #         'batch_spec':[0.5, *[seconds_in_month for _ in range(200)] ],
    #         'divide_by':'time_window',
    #     },
    #     'val':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_month for _ in range(200)],
    #         'divide_by':'time_window',
    #     },
    #     'test':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_month for _ in range(200)],
    #         'divide_by':'time_window',
    #     },
    #    'description': 'Yelp dataset; Train, val and test are batched monthly.',
    #    'big_batch_included':False,
    # },
    # 'wikipedia  50%+daily':{
    #     'train':{
    #         'batch_size':None,
    #         'batch_spec':[0.5, *[seconds_in_day for _ in range(2000)] ],
    #         'divide_by':'time_window',
    #     },
    #     'val':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'test':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'description': 'Wikipedia dataset; training dataset is batched in the following fashion: first batch contains half of the whole train sample, the rest are batched daily. Val and test are batched daily.',
    #     'big_batch_included':True,
    #     'data_params':{
    #         'randomize_features':False,
    #         'fetcher':'transact',
    #         'dataset_name':'wikipedia',
    #         }
    # },
    # 'wikipedia daily':{
    #     'train':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'val':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'test':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'description': 'Wikipedia dataset; Train, val and test are batched daily.',
    #     'big_batch_included':False,
    #     'data_params':{
    #         'randomize_features':False,
    #         'fetcher':'transact',
    #         'dataset_name':'wikipedia',
    #         },
    #
    # },
    # 'reddit  50%+daily':{
    #     'train':{
    #         'batch_size':None,
    #         'batch_spec':[0.5, *[seconds_in_day for _ in range(2000)] ],
    #         'divide_by':'time_window',
    #     },
    #     'val':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'test':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'description': 'Reddit dataset; training dataset is batched in the following fashion: first batch contains half of the whole train sample, the rest are batched daily. Val and test are batched daily.',
    #     'big_batch_included':True,
    #     'data_params':{
    #         'randomize_features':False,
    #         'fetcher':'transact',
    #         'dataset_name':'reddit',
    #         },
    #
    # },
    # 'reddit daily':{
    #     'train':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'val':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'test':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'description': 'Reddit dataset; Train, val and test are batched daily.',
    #     'big_batch_included':False,
    #     'data_params':{
    #         'randomize_features':False,
    #         'fetcher':'transact',
    #         'dataset_name':'reddit',
    #         },
    # },
    # 'yelp_sample weekly':{
    #     'train':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day*7 for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'val':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day*7 for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'test':{
    #         'batch_size':None,
    #         'batch_spec':[seconds_in_day*7 for _ in range(2000)],
    #         'divide_by':'time_window',
    #     },
    #     'description': 'Yelp dataset; Train, val and test are batched weekly.',
    #     'big_batch_included':False,
    #     'data_params':{
    #         'randomize_features':False,
    #         'fetcher':'transact',
    #         'dataset_name':'yelp_sample',
    #         },
    # },
    'uci 1perc':{
        'train':{
            'batch_size':0.01,
            'divide_by':'edge_ix',
        },
        'val':{
            'batch_size':0.01,
            'divide_by':'edge_ix',
        },
        'test':{
            'batch_size':0.01,
            'divide_by':'edge_ix',
        },
        'description': 'UCI dataset; Train, val and test are batched as 100 equal-sized batches.',
        'big_batch_included':False,
        'data_params':{
            'randomize_features':False,
            'fetcher':'transact',
            'dataset_name':'uci',
            },
    },
    'enron 1perc':{
        'train':{
            'batch_size':0.01,
            'divide_by':'edge_ix',
        },
        'val':{
            'batch_size':0.01,
            'divide_by':'edge_ix',
        },
        'test':{
            'batch_size':0.01,
            'divide_by':'edge_ix',
        },
        'description': 'Enron dataset; Train, val and test are batched as 100 equal-sized batches.',
        'big_batch_included':False,
        'data_params':{
            'randomize_features':False,
            'fetcher':'transact',
            'dataset_name':'enron',
            },
    },
    'eth_data_300k 1perc':{
        'train':{
            'batch_size':0.01,
            'divide_by':'edge_ix',
        },
        'val':{
            'batch_size':0.01,
            'divide_by':'edge_ix',
        },
        'test':{
            'batch_size':0.01,
            'divide_by':'edge_ix',
        },
        'description': 'Ethereum dataset; Train, val and test are batched as 100 equal-sized batches.',
        'big_batch_included':False,
        'data_params':{
            'randomize_features':False,
            'fetcher':'transact',
            'dataset_name':'eth_data_300k',
            'bipartite':False,
            },
 }
}


base_experimental = {
    'model':TGNCAWWrapper,
    'data_mode':'transaction', 
    'init_params':{
        'config':{
            'force_cpu':False, 
            'n_epoch':30,
            'n_layer':1,
            'use_caw':True,
            'caw_layers': 1,
            'caw_neighbors':['8', '2'],
            'use_caw_lstm': False,
    }}}
experimental_models={}
# experimental_params = {
#     'caw_neighbors': [['32','1'],['8', '2']],
#     'pos_dim': [10, 100],
#     'use_caw_lstm':[True,False],
#     'use_caw_embed': [True,False],
#     'use_caw_message':[True,False]
#     }
experimental_params = {
    'caw_neighbors': [['32','1'],['8', '2']],
    'pos_dim': [20],
    'use_caw_lstm':[True,False],
    'use_caw_embed': [True],
    'use_caw_message':[False]
    }
for param_set in dict_product(experimental_params):
    set_name ='&&'.join([f'{k}={v}' for k,v in param_set.items()])
    new_config = {**base_experimental['init_params']['config'], **param_set}
    model_spec = deepcopy(base_experimental)
    model_spec['init_params']['config']=new_config
    experimental_models[set_name] = model_spec



base_models = {
    # 'TGN':{'model':TGNWrapper, 'init_params':{'config':{'force_cpu':False, 'n_epoch'  :30}}, 'data_mode':'transaction'},
    # 'Jodie':{'model':TGNWrapper, 'init_params':{'config':{'force_cpu':False, 'n_epoch':30, 'memory_updater':'rnn', 'embedding_module':'time'}}, 'data_mode':'transaction'},
    # 'DyRep':{'model':TGNWrapper, 'init_params':{'config':{'force_cpu':False, 'n_epoch':30, 'memory_updater':'rnn', 'use_destination_embedding_in_message':True}}, 'data_mode':'transaction'},
    # 'DefaultTDGNN': {'model': TDGNNWrapper, 'init_params': {'config': None}, 'data_mode':'transaction'},
    # 'CAW': {'model': CAWWrapper, 'data_mode':'transaction', 'init_params':{'config':{'n_epoch':1, 'force_cpu':False, 'n_layer':1}, }}
    'TiGNNWrapper': {'model': TiGNNWrapper, 'data_mode':'transaction', 'init_params': {'config': {'n_epoch': 20, 'force_cpu':True}}, }
}

all_models = {
    **base_models,
    **experimental_models
}


def run_all_scenarios():
    run_name = f'Run at {time.ctime()}'.replace(':', '_')
    logger = log_utils.get_logger(run_name)
    total_results = {}
    skip_big_batch_settings=True
    i=0
    total_count=len(base_models) * len(batch_param_settings) * len(split_param_settings) * len(node_mask_settings) * len(edge_mask_settings)
    for model_name, model_spec in base_models.items():
        for batch_param_name, batch_params_ in batch_param_settings.items():
            # if skip_big_batch_settings&batch_params_['big_batch_included']:
            #     continue
            data_name = batch_param_name.partition(' ')[0]
            batch_params = {param_name: {**param, 'data_mode':model_spec['data_mode']} for param_name,param in batch_params_.items() if not (param_name in ['description', 'big_batch_included', ])}
            for split_name, split_params in split_param_settings.items():
                for node_mask_name, node_mask_value in node_mask_settings.items():
                    for edge_mask_name, edge_mask_value in edge_mask_settings.items():
                        count = f'{i}_of_{total_count}'
                        name = '__'.join([count, model_name, batch_param_name, split_name, node_mask_name, edge_mask_name])
                        logger.info(f'Starting {i}/{total_count} {name}')
                        try:
                            model = model_spec['model'](**model_spec['init_params'], model_id=name)
                            results = run_scenario(
                                model=model, 
                                dataset_name=data_name, 
                                batch_params=batch_params, 
                                split_params=split_params, 
                                node_mask_frac=node_mask_value, 
                                edge_mask_frac=edge_mask_value,)
                            total_results[name] = results
                            with open(os.path.join('./logs',run_name+'.pkl'),'wb') as f:
                                pkl.dump(file=f, obj=total_results)
                        except Exception as e:
                            logger.error(f'Error! {traceback.format_exc()}')
                        finally:
                            i+=1
                    


# if __name__=='__main__':
#     run_all_scenarios()





# def make_task_runner(logger, total_count):
#     def run_single_task(i, config, logger, total_count):
#         # (
#         #     (model_name, model_spec),
#         #     (batch_param_name, batch_params_),
#         #     (split_name, split_params),
#         #     (node_mask_name, node_mask_value),
#         #     (edge_mask_name, edge_mask_value)
#         # ) = config

#         data_name = batch_param_name.partition(' ')[0]
#         batch_params = {param_name: {**param, 'data_mode':model_spec['data_mode']} for param_name,param in batch_params_.items() if not (param_name in ['description', 'big_batch_included'])}
#         count = f'{i}_of_{total_count}'
#         name = '__'.join([count, model_name, batch_param_name, split_name, node_mask_name, edge_mask_name])
#         logger.info(f'Starting {i}/{total_count} {name}')
#         try:
#             model = model_spec['model'](**model_spec['init_params'], model_id=name)
#             results = run_scenario(
#                 model=model, 
#                 dataset_name=data_name, 
#                 batch_params=batch_params, 
#                 split_params=split_params, 
#                 node_mask_frac=node_mask_value, 
#                 edge_mask_frac=edge_mask_value,)
#         except Exception as e:
#             logger.error(f'Error! {traceback.format_exc()}')
#     return run_single_task



experimental_settings= list(enumerate(itertools.product(
    all_models.keys(),
    # ['wikipedia daily',  'reddit daily', 'enron 1perc',  'uci 1perc'],
    ['eth_data_300k 1perc'],
    ["split 70-10-20"],
    ['10%_label_mask','75%_label_mask'],
    ['10%_edge_mask','75%_edge_mask'],
    ['t','i'],
)))


def make_all_configs(task_configs=None):
    if task_configs is None:
        task_configs = list(enumerate(itertools.product(
            base_models.keys(),
            batch_param_settings.keys(),
            split_param_settings.keys(),
            node_mask_settings.keys(),
            edge_mask_settings.keys()
            )))
    if not os.path.isdir(configs_dir):
        os.mkdir(configs_dir)
    for count in range(10):
        for i, data in task_configs:
            with open(os.path.join(configs_dir, f'{i+count*len(task_configs)} config.txt'), 'w') as f:
                for line in data:
                    f.write(line+'\n')
    

def run_one_batch(i, model_name,batch_param_name,split_name,node_mask_name,edge_mask_name,t_i_setting, run_name, total=None):
    model_spec = all_models[model_name]
    batch_params_ = batch_param_settings[batch_param_name]
    split_params = split_param_settings[split_name]
    node_mask_value = node_mask_settings[node_mask_name]
    edge_mask_value = edge_mask_settings[edge_mask_name]

    logger = log_utils.get_logger(run_name)
    if total is None:
        total_count=len(base_models) * len(batch_param_settings) * len(split_param_settings) * len(node_mask_settings) * len(edge_mask_settings)
    else:
        total_count=total

    data_name = batch_param_name.partition(' ')[0]
    batch_params = {param_name: {**param, 'data_mode':model_spec['data_mode']} for param_name,param in batch_params_.items() if not (param_name in ['description', 'big_batch_included'])}
    count = f'{i}_of_{total_count}'
    name = '__'.join([count, model_name, batch_param_name, split_name, node_mask_name, edge_mask_name])
    logger.info(f'Starting {i}/{total_count} {name}')

    try:
        model = model_spec['model'](**model_spec['init_params'], model_id=name)
        results = run_scenario(
            model=model, 
            dataset_name=data_name, 
            batch_params=batch_params, 
            split_params=split_params, 
            node_mask_frac=node_mask_value, 
            edge_mask_frac=edge_mask_value,
            t_i_setting=t_i_setting)
    except Exception as e:
        logger.error(f'Error on {name}! \n{traceback.format_exc()}')

def read_batch_config(filename):
    with open(os.path.join(configs_dir, filename), 'r') as f:
        data = [x.strip() for x in f.readlines()]
    return data

def process_one_run():
    i = sys.argv[1]
    run_name = os.environ.get('RUN_NAME', 'noname')
    filename = [x for x in os.listdir(configs_dir) if x.startswith(str(i)+' ')][0]
    total = len(os.listdir(configs_dir)) 
    model_name,batch_param_name,split_name,node_mask_name,edge_mask_name, t_i_setting = read_batch_config(filename)
    run_one_batch(i, model_name,batch_param_name,split_name,node_mask_name,edge_mask_name, t_i_setting, run_name, total)


if __name__=='__main__':
    if len(sys.argv)>1:
        process_one_run()
    else:
        run_all_scenarios()