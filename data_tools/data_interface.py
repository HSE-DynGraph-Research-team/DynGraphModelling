import os
import numpy as np
from .TGN_data_obj import Data as TGNData
from torch_geometric.data import Data as PGData
import warnings
import torch
import bisect

default_node_features_dim = 172


def drop_duplicates_keep_last(arr):
    rev_ix = np.unique(arr[::-1], return_index=True)
    ix = len(arr)-rev_ix[-1]-1
    return ix


class NodeFeatures:
    """
    Dispenser for node features, which can be indexed by node id and timestamp, compatible with base TGN implementation
    """

    def __init__(self, node_table, total_n_nodes):

#         self.node_dict = {i: node_table[node_table[:,0]==i] for i in tqdm(np.unique(node_table[:,0]))}

        self.node_dict = {i: np.zeros((1, node_table.shape[1])) for i in np.unique(node_table[:,0])}
        for ix, i in enumerate(np.unique(node_table[:,0])):
            self.node_dict[i] = np.vstack((self.node_dict[i], node_table[ix].reshape((1, node_table.shape[1]))))
        for key, val in self.node_dict.items():
            self.node_dict[key] = val[1:]
        self.total_n_nodes = total_n_nodes
        self.shape = (self.total_n_nodes, node_table.shape[1]-2)


    def __getitem__(self, key):
        feats = []
        key = [k.cpu().numpy() if isinstance(k, torch.Tensor) else k for k in key]
        arr_list = [[self.node_dict[node], ts] for node, ts in zip(*key)]
        feats = [arr[np.where(arr[:,1]<=ts)[0][-1],2:] for arr, ts in  arr_list]
        return torch.Tensor(np.array(feats)).float()

    def get(self):
        return np.stack([self.node_dict[i][0,2:] if i in self.node_dict else np.zeros(self.shape[1]) for i in range(self.total_n_nodes)],axis=0) 

class EdgeFeatures:
    """
    Dispenser for edge features, which can be indexed by edge_id, compatible with base TGN implementation
    """

    def __init__(self, edge_table):
        self.edge_table = edge_table
        self.shape = (self.edge_table.shape[0], self.edge_table.shape[1]-4)

    def __getitem__(self, key):
        feats = []
        if isinstance(key,tuple):
            ids = key[0]
        else:
            ids=key
        if isinstance(ids, int):
            ids = [ids]
        elif isinstance(ids, float):
            ids = [int(ids)]
        elif isinstance(ids, np.ndarray):
            ids = ids.astype(int)
        elif isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy().astype(int)
        return torch.Tensor(self.edge_table[np.array(ids).astype(int), 4:]).float() 

    def get(self):
        return self.edge_table[:,4:]


class GraphContainer(): 

    """
    Internal representation of data consists of three arrays:

    node_table:
        node_id
        last_updated
        features

    edge table - as was:
        edge_id (unique)
        timestamp
        s_node_id
        d_node_id
        edge_features

    labels:
        node_id
        timestamp
        state
    """

    def __init__(
        self, 
        data=None, 
        n_unique_nodes=None,
        ):
        if data is None:
            warnings.warn('Fill with data with either "read_transact_data" or "read_snapshot_data"')
        else:
            edge_table,node_table,label_table = data
            self.edge_table = edge_table
            self.node_table = node_table
            self.label_table = label_table
            self.timesteps = self.edge_table[:,1]
            self.timesteps_are_unique = np.unique(self.timesteps).shape[0]==self.timesteps.shape[0]
            self.n_interactions = self.edge_table.shape[0]
            self.unique_nodes = np.unique(self.edge_table[:,[2,3]].reshape(-1))
            self.n_unique_nodes = len(self.unique_nodes)
            if n_unique_nodes is None:
                raise ValueError('Provide total unique node count')
            else:
                self.total_n_nodes = n_unique_nodes
        self.edge_dispenser = None
        self.node_dispenser = None
            

    @property
    def edge_features(self):
        if self.edge_dispenser is None:
            self.edge_dispenser = EdgeFeatures(self.edge_table)
        return self.edge_dispenser

    @property
    def node_features(self):
        if self.node_dispenser is None:
            self.node_dispenser = NodeFeatures(self.node_table, self.total_n_nodes)
        return self.node_dispenser

    def read_transact_data(self, data_obj, dynamic_node_features=False):
        if dynamic_node_features:
            data_obj_, source_node_features, destination_node_features, edge_features = data_obj
        else:
            data_obj_, node_features, edge_features = data_obj

        self.edge_table = np.concatenate([ np.expand_dims(x,1) if np.ndim(x)==1 else x for x in (
            np.arange(len(data_obj_.timestamps))+1, #start edge ixes from 1
            data_obj_.timestamps,
            data_obj_.sources, 
            data_obj_.destinations,
            edge_features
            )],axis=1)
        self.edge_table = np.concatenate([
            np.zeros((1,self.edge_table.shape[1])), 
            self.edge_table],axis=0)

        if dynamic_node_features:    
            source_node_table = np.concatenate([
                data_obj_.sources,
                data_obj_.timestamps,
                source_node_features
            ], axis=1)
            destination_node_table = np.concatenate([
                data_obj_.destinations,
                data_obj_.timestamps,
                destination_node_features
            ], axis=1)
            node_temp = np.concatenate([ source_node_table,destination_node_table],axis=0)

            #sort by timestamp, then by node id
            self.node_table = node_temp[np.lexsort((node_temp[:,1],node_temp[:,0]))]

        else:
            #node features are static, timestep is 0
            self.node_table = np.concatenate([
                np.arange(node_features.shape[0]).reshape(-1,1)+1,
                np.zeros(shape=(node_features.shape[0],1)),
                node_features
                ], axis=1)
            self.node_table = np.concatenate([np.zeros((1,self.node_table.shape[1])), self.node_table],axis=0)
        self.label_table = np.concatenate([
            data_obj_.sources.reshape(-1,1), 
            data_obj_.timestamps.reshape(-1,1), 
            data_obj_.labels.reshape(-1,1), 
            ], axis=1)

        self.label_table = np.concatenate([
            np.zeros((1,self.label_table.shape[1])), 
            self.label_table],axis=0)


        sorted_edges = np.argsort(self.edge_table[:,1])
        self.edge_table = np.concatenate([
            np.arange(self.edge_table.shape[0]).reshape(-1,1),
            self.edge_table[sorted_edges,1:]
        ],axis=1)

        self.label_table = self.label_table[sorted_edges]

        self.timesteps = self.edge_table[:,1]
        self.timesteps_are_unique = np.unique(self.timesteps).shape[0]==self.timesteps.shape[0]
        self.n_interactions = self.edge_table.shape[0]
        self.unique_nodes = np.unique(self.edge_table[:,[2,3]].reshape(-1))
        self.n_unique_nodes = len(self.unique_nodes)
        self.total_n_nodes = np.unique(self.node_table[:,0]).shape[0]

    def read_snapshot_data(self, data_obj):
        
        if hasattr(data_obj, 'edge_index'):
            dynamic_edges = False
        elif hasattr(data_obj, 'edge_indices'):
            dynamic_edges = True
            tsteps = len(data_obj.edge_indices)
        else:
            raise ValueError('Bad object')

        if hasattr(data_obj, 'feature'):
            dynamic_nodes = False
        elif hasattr(data_obj, 'features'):
            dynamic_nodes = True
            tsteps = len(data_obj.features)
        else:
            raise ValueError('Bad object')

        if dynamic_edges:
            numerated_edges = [np.concatenate(
                (
                    np.ones((len(edges), 1))*i, 
                    np.array(edges)
                    ),axis=1
                ) for i, edges in enumerate([x.T for x in data_obj.edge_indices])]

            edge_weights = np.concatenate(data_obj.edge_weights, axis=0)
            if np.ndim(edge_weights)==1:
                edge_weights = edge_weights.reshape(-1,1)

            self.edge_table = np.concatenate((
                np.concatenate(numerated_edges, axis=0), 
                edge_weights, 
                ),axis=1)

        else:
            if np.ndim(data_obj.edge_weight)==1:
                edge_weight = data_obj.edge_weight.reshape(-1,1)
            else:
                edge_weight = data_obj.edge_weight
            assert edge_weight.shape[0]==data_obj.edge_index.T.shape[0]

            self.edge_table = np.concatenate([np.concatenate([
                np.ones((edge_weight.shape[0], 1))*i,
                data_obj.edge_index.T,
                edge_weight
            ],axis=1) for i in range(tsteps)], axis=0)

        self.edge_table = np.concatenate([
            np.arange(self.edge_table.shape[0]).reshape(-1,1),
            self.edge_table
        ], axis=1)


        if dynamic_nodes:
            numerated_nodes = [np.concatenate((
                np.arange(len(nodes)).reshape(-1, 1),
                np.ones((len(nodes), 1))*i,
                np.array(nodes)
            ),axis=1) for i, nodes in enumerate(data_obj.features)]
            self.node_table = np.concatenate(numerated_nodes,axis=0)

        else:
            self.node_table = np.concatenate([
                np.arange(data_obj.feature.shape[0]).reshape(-1,1),
                np.zeros((data_obj.feature.shape[0],1)),
                data_obj.feature],axis=1)
            

        self.label_table = np.concatenate(
            [np.concatenate([np.arange(len(targets)).reshape(-1,1),np.ones((len(targets),1))*i, targets.reshape(-1,1)],axis=1)
            for i, targets in enumerate(data_obj.targets)], axis=0)

        self.timesteps = self.edge_table[:,1]
        self.timesteps_are_unique = np.unique(self.timesteps).shape[0]==self.timesteps.shape[0]
        self.n_interactions = self.edge_table.shape[0]
        self.unique_nodes = np.unique(self.edge_table[:,[2,3]].reshape(-1))
        self.n_unique_nodes = len(self.unique_nodes)
        self.total_n_nodes = np.unique(self.node_table[:,0]).shape[0]        

    def read_from_source(self, data_obj, kind):
        if kind=='snapshot':
            pass
        elif kind=='transaction':
            pass
        else:
            raise ValueError(f'Unexpected value {str(kind)}; valid options - {{"snapshot", "transaction"}}')
        return None

    def get_edge_ix_by_ts(self, ts):
        return self.edge_table[self.edge_table[:,1]==ts, 0]

    def get_ts_by_edge_ix(self, edge_ix):
        return self.edge_table[self.edge_table[:,0]==edge_ix, 1][0]

    def val_split_by_time(self, fractions=[0.7,0.2,0.1]):
        fracs = np.cumsum(np.array([0]+fractions)*(1/sum(fractions)))
        borders = (fracs *self.edge_table.shape[0]).astype(int)
        assert np.isclose(fracs[-1],1)
        assert borders[-1]==self.edge_table.shape[0], f'{borders[-1]}!={self.edge_table.shape[0]}'
        borders[-1] -=1
        ts_borders = [self.get_ts_by_edge_ix(x) for x in borders]
        for i in range(len(borders)-1):
            nodes_before = self.node_table[self.node_table[:,1]<(ts_borders[i] if i else 1)]
            nodes_before = nodes_before[drop_duplicates_keep_last(nodes_before[:,0])]
            nodes_between = self.node_table[(self.node_table[:,1]>ts_borders[i])&(self.node_table[:,1]<=ts_borders[i+1])]
            node_data = np.concatenate([nodes_before, nodes_between],axis=0)

            labels_before = self.label_table[self.label_table[:,1]<(ts_borders[i] if i else 1)]
            labels_before = labels_before[drop_duplicates_keep_last(labels_before[:,0])]
            labels_between = self.label_table[(self.label_table[:,1]>ts_borders[i])&(self.label_table[:,1]<=ts_borders[i+1])]
            label_data = np.concatenate([labels_before, labels_between],axis=0)
            data = (
                self.edge_table[borders[i]:borders[i+1]],
                node_data,
                label_data,
                )
            
            yield GraphContainer(data=data, n_unique_nodes=self.total_n_nodes)
        
    def __iter__(self):
        for batch_ixes in self.batch_bounds:
            batch = self.batcher(batch_ixes)
            yield batch

    def __call__(
        self, 
        *args, 
        data_mode='transaction',
        batch_size=None,
        batch_spec=None,
        divide_by=None,
        **kwargs):
        """
        Generate batches by specification
        Specify either batch_size or batch_spec (both accept ints for counts and floats for fractions)

        data_mode: {'transaction','snapshot'} - format to return batches with
        batch_size: [int or float] - int for batchsize, float for fraction
        batch_spec: [list of floats or list of ints] - specification for generating all of the batches
        divide_by: {'edge_ix','time_stamp','time_window'} - what to use as a basis to divide into batches:
            divide_by=='edge_ix' - batch by individual edges (with batch_size=1 one batch contains one edge)
            divide_by=='time_stamp' - batch by unique timestamps (with batch_size=1 one batch contains data from one timestamp - full snapshot in snapshot data, one edge in transact-like data)
            divide_by=='time_window' - batch by time windows (batch_size=0.1 equals exactly 10% of time between min and max of available time;)
        """


        assert divide_by in ['edge_ix','time_stamp','time_window'], 'Wrong value for "divide_by"' 
        # relying on the fact that 1st edge_table col is edge_ix, and 2nd is time
        steps_list = np.unique(self.edge_table[:,int(divide_by.startswith('time'))])
        if divide_by=='time_window':
            steps_list = np.linspace(min(steps_list),max(steps_list),int(max(steps_list)-min(steps_list)+1))

        if not (batch_size is None):
            if isinstance(batch_size,float):
                batch_size = int(batch_size*len(steps_list))
            steps = [[steps_list[i*batch_size], steps_list[min(len(steps_list)-1,(i+1)*batch_size)]] for i in range((len(steps_list)//batch_size))]
        
        elif not (batch_spec is None):
            # transform all specs to fractions
            frac_batch_spec1 = [x/len(steps_list) if x>1 else x for x in batch_spec]
            # trim the excess batches
            cumsum = np.cumsum(frac_batch_spec1)
            frac_batch_spec2 = [frac_batch_spec1[i] for i in range(len(frac_batch_spec1)) if cumsum[i]<1]
            # normalize and add 0
            ixes = np.concatenate([[0], np.cumsum(frac_batch_spec2)])*1/sum(frac_batch_spec2)
            # bring to actual indices
            ixes = [int(x*len(steps_list)) for x in ixes]
            steps = [[ steps_list[min(len(steps_list)-1,ixes[i])], steps_list[min(len(steps_list)-1, ixes[i+1])]] for i in range(len(ixes)-1)]
        else:
            raise ValueError('Specify at least one of {"batch_size","batch_spec"}')

        # steps is list of lists; each sublist indicates corner values for each batch on respective axis (edge ix or timestamp)

        pre_batches = [np.array([]) for _ in range(len(steps))]

        step_min = float('inf')
        step_max = float('-inf')
        for step in steps:
            val = step[1] - step[0]
            step_min = min(step_min, val)
            step_max = max(step_max, val)

        map_id_to_ix = {}
        for i, row in enumerate(self.edge_table):
            choose_col = int(divide_by.startswith('time'))
            ix_min = min(int(row[choose_col] /(step_max + 1e-5)), len(steps) - 1)
            ix_max = min(int(row[choose_col] /(step_min + 1e-5)) + 1, len(steps))
            
            if row[int(divide_by.startswith('time'))] >= steps[-1][1]:
                continue
            ix = bisect.bisect_left(steps, [row[int(divide_by.startswith('time'))], float('inf')], ix_min, ix_max) - 1
            pre_batches[ix] = np.append(pre_batches[ix], row[0])
            map_id_to_ix[row[0]] = i

        self.batch_bounds = [[map_id_to_ix[int(x[0])], map_id_to_ix[int(x[-1])]] for x in pre_batches if len(x)>0]
        self.num_batches = len(self.batch_bounds)

        if data_mode=='snapshot':
            self.batcher = self.get_batch_snapshot
        elif data_mode=='transaction':
            self.batcher = self.get_batch_transaction
        else:
            raise ValueError(f"Invalid value of 'data_mode' ({data_mode}), expected 'snapshot' or 'transaction'")
        return self

    def get_batch_snapshot(self, batch_bounds):

        s_ix, e_ix = batch_bounds
        s_ts =  self.edge_table[s_ix, 1]
        e_ts =  self.edge_table[e_ix, 1]

        node_feat = self.node_table[self.node_table[:,1]<=e_ts]
        x = node_feat[drop_duplicates_keep_last(node_feat[:,0]), 2:]
        edge_index = self.edge_table[s_ix:e_ix, [2,3]].astype(int)
        edge_attr =  self.edge_table[s_ix:e_ix, 4:]
        y = self.label_table[self.label_table[:, 1]<=e_ts]
        y = y[drop_duplicates_keep_last(y[:,0]), 2:]
        
        snapshot = PGData(
            x = x,
            edge_index = edge_index,
            edge_attr = edge_attr,
            y = y)
        return snapshot

    def get_total_transaction(self):
        sources_total      = self.edge_table[ :,  2]
        destinations_total = self.edge_table[ :,  2]
        timestamps_total   = self.edge_table[ :,  1]
        edge_idxs_total    = self.edge_table[ :,  0]
        labels_total       = self.label_table[:, -1]
        return sources_total, destinations_total, timestamps_total, edge_idxs_total, labels_total

    # def find_current_label(self, node_id, ts):
    #     if getattr(self, 'label_dict', None) is None:
    #         self.label_dict = {node: self.label_table[self.label_table[:,0]==node_id] for node in np.unique(self.label_table[:,0])}
    #     return self.label_dict[node_id][self.label_dict[node_id][:,1]<=ts][-1,2]
        # return self.label_table[(self.label_table[:,0]==node_id)&(self.label_table[:,1]<=ts), 2][-1]

    def get_batch_transaction(self, batch_bounds):
        s_ix, e_ix         = batch_bounds
        sources_batch      = self.edge_table[ s_ix:e_ix,  2].astype(int)
        destinations_batch = self.edge_table[ s_ix:e_ix,  3].astype(int)
        timestamps_batch   = self.edge_table[ s_ix:e_ix,  1]
        edge_idxs_batch    = self.edge_table[ s_ix:e_ix,  0].astype(int)
        # labels_batch       = np.array([self.find_current_label(id_, ts) for id_, ts in zip(sources_batch, timestamps_batch)])
        labels_batch       = self.label_table[s_ix:e_ix, 2]
        return sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, labels_batch

    def partition_snapshot_data_by_mask(self, mask):
        """
        Provided collection of boolean mask on edges table, create partitions of original data
        """

        assert mask.shape[0]==self.edge_table.shape[0]
        new_edge_table = self.edge_table[mask]
        ts_s, ts_e = min(new_edge_table[:,1]),max(new_edge_table[:,1])

        nodes_before = self.node_table[self.node_table[:,1]<(ts_s)]
        nodes_before = nodes_before[drop_duplicates_keep_last(nodes_before[:,0])]
        nodes_between = self.node_table[(self.node_table[:,1]>ts_s)&(self.node_table[:,1]<=ts_e)]
        node_data = np.concatenate([nodes_before, nodes_between],axis=0)

        # node_data = self.node_table[mask]

        # labels_before = self.label_table[self.label_table[:,1]<(ts_s)]
        # labels_before = labels_before[drop_duplicates_keep_last(labels_before[:,0])]
        # labels_between = self.label_table[(self.label_table[:,1]>ts_s)&(self.label_table[:,1]<=ts_e)]
        # label_data = np.concatenate([labels_before, labels_between],axis=0)
        label_data = self.label_table[mask]

        return GraphContainer(
            data=(
                new_edge_table,
                node_data,
                label_data,
                ),
            n_unique_nodes=self.total_n_nodes
        )

def integrate_sampled_destinations_into_transact_batch(transact_data, sampled):
    raise NotImplementedError

def integrate_sampled_destinations_into_snapshot_batch(snapshot_data, sampled):
    raise NotImplementedError


if __name__=='__main__':
    import torch_geometric_temporal as tgt
    from collections import Counter
    DGTS = tgt.dataset.encovid.EnglandCovidDatasetLoader()# DynamicGraphTemporalSignal
    SGTS = tgt.dataset.chickenpox.ChickenpoxDatasetLoader() #StaticGraphTemporalSignal
    data_obj_DGTS = DGTS.get_dataset()
    data_obj_SGTS = SGTS.get_dataset()
    print('Dataset loaded')
    for data_obj in [data_obj_DGTS,data_obj_SGTS]:
        GC = GraphContainer()
        GC.read_snapshot_data(data_obj)
        train,val,test = list(GC.val_split_by_time())
        batches = [batch for batch in GC(batch_spec=[0.1,0.2,0.2,0.3,0.2], divide_by='edge_ix')]
        print(f'{len(batches)} batches, {str(Counter([np.stack(x,axis=1).shape for x in batches]).most_common())}')
        for ds in [train, val, test]:
            batches = [batch for batch in ds(batch_size=2, divide_by='time')]
            print(f'{len(batches)} batches, {str(Counter([np.stack(x,axis=1).shape for x in batches]).most_common())}')

