import numpy as np
import os
import pandas as pd
from .raw_transaction_processing import run as run_preprocess
from .data_interface import GraphContainer
from .TGN_data_obj import Data as TGNData



def prepare_transact_dataset(
    dataset_name, 
    data_path='./data',
    randomize_features=False, 
    **kwargs,
):
    graph_df_path = os.path.join(data_path, f'ml_{dataset_name}.csv')
    edge_features_path = os.path.join(data_path, f'ml_{dataset_name}.npy')
    node_features_path = os.path.join(data_path, f'ml_{dataset_name}_node.npy')
    if not all([os.path.isfile(x) for x in [graph_df_path, edge_features_path, node_features_path]]):
        if os.path.isfile(os.path.join(data_path, dataset_name+'.csv')):
            run_preprocess(dataset_name, data_path, bipartite=kwargs.get('bipartite', True))
        else:
            raise FileNotFoundError(f'Cant find {dataset_name} dataset to preprocess (expected to be in dir "{os.path.abspath(data_path)}")')

    graph_df = pd.read_csv(graph_df_path)
    edge_features = np.load(edge_features_path)
    node_features = np.load(node_features_path)

    if randomize_features:
        node_features = np.random.rand(
            node_features.shape[0], node_features.shape[1])


    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = GraphContainer()
    full_data.read_transact_data(
        (
            TGNData(sources, destinations, timestamps, edge_idxs, labels),
            node_features,
            edge_features,
        )
    )
    return full_data



fetchers = {
    'transact':prepare_transact_dataset,
}
