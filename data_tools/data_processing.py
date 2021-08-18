import numpy as np
import random
import os
import pandas as pd
from .raw_transaction_processing import run as run_preprocess
from .data_interface import GraphContainer
from .TGN_data_obj import Data as TGNData
from .data_fetchers import fetchers






add_zero = lambda mask, add: np.concatenate([np.array([False]),mask]) if add else mask



def prepare_transductive_data_from_transact(
    split_params=None, 
    node_mask_frac=0.1,
    edge_mask_frac=0.1,
    seed=2020,# np.random.randint(low=0,high=1000),
    data_params={
        'randomize_features':False,
        'data_path':'./data',
        'fetcher':'transact',
        'dataset_name':'wikipedia',
    }
    ):
    # Load data and train val test split
    full_data = fetchers[data_params['fetcher']](**data_params)


    if split_params is None:
        split_params = [0.70, 0.80]

    timestamps = full_data.edge_table[:,1]
    sources = full_data.edge_table[:,2]
    destinations = full_data.edge_table[:,3]



    val_time, test_time = list(np.quantile(timestamps, split_params))

    random.seed(seed)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    #edge masking
    edge_unmasked = np.random.uniform(size=timestamps.shape[0])>edge_mask_frac

    train_mask = (timestamps<=val_time) * edge_unmasked
    #val cant have masked nodes
    val_mask = (timestamps>val_time) * (timestamps<=test_time)
    #transductive test mask - only known nodes
    test_mask = (timestamps>test_time)

    train_data = full_data.partition_snapshot_data_by_mask(add_zero(train_mask, full_data.edge_table.shape[0]!=train_mask.shape[0]))
    # validation and test with all edges
    val_data = full_data.partition_snapshot_data_by_mask(add_zero(val_mask, full_data.edge_table.shape[0]!=val_mask.shape[0]))
    # validation and test with edges that at least has one new node (not in training set)
    test_data = full_data.partition_snapshot_data_by_mask(add_zero(test_mask, full_data.edge_table.shape[0]!=test_mask.shape[0]))


    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))
    return full_data.node_features, full_data.edge_features, full_data, train_data, val_data, test_data


def prepare_inductive_data_from_transact(
    split_params=None, 
    node_mask_frac=0.1,
    edge_mask_frac=0.1,
    seed=2020,# np.random.randint(low=0,high=1000),
    data_params={
        'randomize_features':False,
        'data_path':'./data',
        'fetcher':'transact',
        'dataset_name':'wikipedia',
    }
    ):
    # Load data and train val test split
    full_data = fetchers[data_params['fetcher']](**data_params)

    if split_params is None:
        split_params = [0.70, 0.80]

    timestamps = full_data.edge_table[:,1]
    labels = full_data.label_table[:,2]
    sources = full_data.edge_table[:,2]
    destinations = full_data.edge_table[:,3]



    val_time, test_time = list(np.quantile(timestamps, split_params))


    random.seed(seed)

    node_set = set(sources) | set(destinations)
    non_train_node_set = set(sources[timestamps>val_time]) | set(destinations[timestamps>val_time])
    n_total_unique_nodes = len(node_set)
    n_nontrain_unique_nodes = len(non_train_node_set)


    n_unique_labels = len(np.unique(labels))
    while True:
        mask_node_set = set(random.sample(set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time])), int(node_mask_frac * n_nontrain_unique_nodes)))
        mask_source = np.array([x in mask_node_set for x in sources])
        mask_destination = np.array([x in mask_node_set for x in destinations])
        unmasked_mask =  ((1 - mask_source) * (1 - mask_destination)).astype(bool)


        val_mask = (timestamps>val_time) * (timestamps<=test_time) * unmasked_mask
        if len(np.unique(labels[val_mask]))==n_unique_labels:
            break
        else:
            random.seed()
    random.seed(seed)
            

    #edge masking, independent of node masking
    edge_unmasked = np.random.uniform(size=timestamps.shape[0])>edge_mask_frac

    #non-masked edges
    #train can't have masked nodes
    train_mask = (timestamps<=val_time) * unmasked_mask * edge_unmasked
    #val cant have masked nodes
    val_mask = (timestamps>val_time) * (timestamps<=test_time) * unmasked_mask
    #transductive test mask - only known nodes
    # test_transductive_mask = (timestamps>test_time) * unmasked_mask
    #total inductive mask - all edges with any masked node
    test_inductive_all_mask = (timestamps>test_time) * (1-unmasked_mask).astype(bool)
    #strict inductive mask - both nodes of an edge are masked
    test_both_nodes_inductive_mask = (timestamps>test_time) * mask_source * mask_destination
    #lean inducive mask - exactly one node masked
    test_one_node_inductive_mask = (test_inductive_all_mask.astype(int) - test_both_nodes_inductive_mask.astype(int)).astype(bool)



    train_data = full_data.partition_snapshot_data_by_mask(add_zero(train_mask, full_data.edge_table.shape[0]!=train_mask.shape[0]))
    # validation and test with all edges
    val_data = full_data.partition_snapshot_data_by_mask(add_zero(val_mask, full_data.edge_table.shape[0]!=val_mask.shape[0]))
    # validation and test with edges that at least has one new node (not in training set)
    test_data = full_data.partition_snapshot_data_by_mask(add_zero(test_one_node_inductive_mask, full_data.edge_table.shape[0]!=test_one_node_inductive_mask.shape[0]))


    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The inductive test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))
    return full_data.node_features, full_data.edge_features, full_data, train_data, val_data, test_data


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(
            c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
