import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Union
import re
import pickle as pkl


def preprocess(
    data_name, 
    node_feat: Union[None, int],
    node_feat_which='both',
    ):
    """
    data_name: str, name of raw dataset (to be found in data folder)
    node_feat: {None, int}, number of node features in each row, None if no node features
    node_feat_which: {'source','destination','both'}, specify which node features are present in each row  
    """
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_edge = []
    feat_node = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])

            if node_feat:
                if node_feat_which=='both':
                    
                    feat_e = np.array([float(x) for x in e[4:-node_feat*2]])
                    feat_n_s = np.array([u, ts ] + [float(x) for x in e[-node_feat*2:-node_feat]])
                    feat_n_d = np.array([i, ts ] + [float(x) for x in e[-node_feat:]])
                    feat_n = [feat_n_s, feat_n_d]
                else:
                    feat_e = np.array([float(x) for x in e[4:-node_feat]])
                    if node_feat_which=='source':
                        feat_n = np.array([u, ts ] + [float(x) for x in e[-node_feat:]])
                    elif node_feat_which=='destination':
                        feat_n = np.array([i, ts ] + [float(x) for x in e[-node_feat:]])
                    else:
                        raise ValueError(f"Expected {{'source', 'destination', 'both'}} in 'node_feat_which' argument, got {node_feat_which}")
            else:
                feat_e = np.array([float(x) for x in e[4:]])
                feat_n = []
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_edge.append(feat_e)
            feat_node.append(feat_n)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_edge), np.array(feat_node)


def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite:

        u_map = {old_id:new_id for new_id,old_id in enumerate(df.u.unique())}
        i_map = {old_id:new_id for new_id,old_id in enumerate(df.i.unique())}

        new_df.u = new_df.u.apply(lambda x: u_map[x])
        new_df.i = new_df.i.apply(lambda x: i_map[x])

        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
    else:
        u_map = {old_id:new_id for new_id,old_id in enumerate(np.unique(new_df[['u','i']].values.reshape(-1)))}
        i_map = None
        new_df.u = new_df.u.apply(lambda x: u_map[x])
        new_df.i = new_df.i.apply(lambda x: u_map[x])


    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    return new_df, u_map, i_map

def get_node_params(filename):
    pat = 'node([sdb])=(\d+)'
    translate_type = {
        'b':'both',
        's':'source',
        'd':'destination',
    }
    params = re.findall(pat, filename)
    if params:
        node_feat_which, node_feat  = params[0]
        node_feat_which = translate_type[node_feat_which]
        node_feat = int(node_feat)
    else:
        node_feat, node_feat_which = None, ''

    return node_feat, node_feat_which


def run(dataset_name, data_path='./data', bipartite=True, node_feat = False):
    node_feat, node_feat_which = get_node_params(dataset_name)
    PATH = os.path.join(data_path, f'{dataset_name}.csv')
    OUT_DF = os.path.join(data_path, f'ml_{dataset_name}.csv')
    OUT_FEAT = os.path.join(data_path, f'ml_{dataset_name}.npy')
    OUT_NODE_FEAT = os.path.join(data_path, f'ml_{dataset_name}_node.npy')
    META_DATA = os.path.join(data_path, f'ml_{dataset_name}_meta.pkl')
    df, feat_edge, feat_node = preprocess(PATH, node_feat, node_feat_which)
    new_df, u_map, i_map = reindex(df, bipartite)

    if not node_feat:
        max_idx = max(new_df.u.max(), new_df.i.max())
        feat_node = np.zeros((max_idx + 1, 172))

    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat_edge)
    np.save(OUT_NODE_FEAT, feat_node)
    with open(META_DATA, 'w') as f:
        pkl.dump(f, (u_map, i_map))

