from models.tgn.tgn_wrapper import TGNWrapper
from models.tgn_new.tgn_wrapper import TGNCAWWrapper
from models.tdgnn.tdgnn_wrapper import TDGNNWrapper
from models.CAW.caw_wrapper import CAWWrapper

model_specs = {
    'TGN':          {'model': TGNWrapper,   'data_mode':'transaction', 'init_params': {'config':{'force_cpu':False, 'n_epoch':30}}},
    'Jodie':        {'model': TGNWrapper,   'data_mode':'transaction', 'init_params': {'config':{'force_cpu':False, 'n_epoch':30, 'memory_updater':'rnn', 'embedding_module':'time'}}},
    'DyRep':        {'model': TGNWrapper,   'data_mode':'transaction', 'init_params': {'config':{'force_cpu':False, 'n_epoch':30, 'memory_updater':'rnn', 'use_destination_embedding_in_message':True}}},
    'DefaultTDGNN': {'model': TDGNNWrapper, 'data_mode':'transaction', 'init_params': {'config': None}},
    'CAW':          {'model': CAWWrapper,   'data_mode':'transaction', 'init_params': {'config':{'n_epoch':1, 'force_cpu':False, 'n_layer':1}, }},
    'CAWTGN':       {'model': TGNCAWWrapper,'data_mode':'transaction', 'init_params': {'config':{ 'force_cpu':False,  'n_epoch':30, 'n_layer':1, 'use_caw':True, 'caw_layers': 1, 'caw_neighbors':['8', '2'], 'use_caw_lstm': False,}}}
    }