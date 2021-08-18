


model_specs={} #import from special place!
import os
import yaml
import sys
from model_specs import model_specs
from pipelines import combined_pipeline

def parse_params(path_to_yaml):
    with open(path_to_yaml) as f:
        doc = yaml.full_load(f)


    final_params = {}
    assert isinstance(doc['node_mask_frac'],float)
    assert (doc['node_mask_frac']>0)&(doc['node_mask_frac']<1)
    final_params['node_mask_frac'] = doc['node_mask_frac']

    assert isinstance(doc['edge_mask_frac'],float)
    assert (doc['edge_mask_frac']>0)&(doc['edge_mask_frac']<1)
    final_params['edge_mask_frac'] = doc['edge_mask_frac']

    assert type(doc['split_params'])==list
    assert all([isinstance(x, float) for x in doc['split_params']])
    assert doc['split_params'][1]>doc['split_params'][0]
    final_params['split_params'] = doc['split_params']

    assert doc['t_i_setting'] in ['t','i','ti', None]
    final_params['t_i_setting'] = doc['t_i_setting']


    assert doc['model']['data_mode'] in ['transaction','snapshot']

    data_mode = doc['model']['data_mode']
    assert data_mode in ['transaction','snapshot']

    model_obj = doc['model']['architecture']
    model_id = doc['model']['model_id']
    model_config = doc['model']['config']

    assert model_obj in model_specs
    assert isinstance(model_config, dict)
    assert isinstance(model_id, str)

    default_model_config = model_specs[model_obj]['init_params']
    final_model_config = {**default_model_config, **model_config}
    model = model_specs[model_obj]['model'](config=final_model_config, model_id=model_id)

    final_params['model_wrapper'] = model


    if any([x is None for x in doc['batch_params']['universal'].values()]):
        batch_params = {
            'train':{**doc['batch_params']['train'], 'data_mode':data_mode},
            'val':{**doc['batch_params']['val'], 'data_mode':data_mode},
            'test':{**doc['batch_params']['test'], 'data_mode':data_mode},
        }
    else:
        batch_params = {
            'train':{**doc['batch_params']['universal'], 'data_mode':data_mode},
            'val':{**doc['batch_params']['universal'], 'data_mode':data_mode},
            'test':{**doc['batch_params']['universal'], 'data_mode':data_mode},
        }
    final_params['batch_params'] = batch_params

    final_params['data_params'] = {**doc['data_params'], 'data_mode':data_mode}

    return final_params



if __name__=='__main__':
    path = sys.argv[1]
    if not os.path.isfile(path):
        #launch on cluster
        directory = os.environ['RUN_NAME']
        path = os.path.join(directory, f'{path}.yaml')
    params = parse_params(path)
    results = combined_pipeline(**params)    