import time
import numpy as np
import torch
import multiprocessing as mp
import torch.nn as nn
from .caw_position import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionEncoder(nn.Module):
    '''
    Note that encoding initialization and lookup is done on cpu but encoding (post) projection is on device
    '''
    def __init__(self, num_layers, enc='spd', enc_dim=2, ngh_finder=None, verbosity=1, cpu_cores=1, logger=None):
        super(PositionEncoder, self).__init__()
        self.enc = enc
        self.enc_dim = enc_dim
        self.num_layers = num_layers
        self.nodetime2emb_maps = None
        self.projection = nn.Linear(1, 1)  # reserved for when the internal position encoding does not match input
        self.cpu_cores = cpu_cores
        self.ngh_finder = ngh_finder
        self.verbosity = verbosity
        self.logger = logger
        if self.enc == 'spd':
            self.trainable_embedding = nn.Embedding(num_embeddings=self.num_layers+2, embedding_dim=self.enc_dim) # [0, 1, ... num_layers, inf]
        else:
            assert(self.enc in ['lp', 'saw'])
            self.trainable_embedding = nn.Sequential(nn.Linear(in_features=self.num_layers+1, out_features=self.enc_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim))  # landing prob at [0, 1, ... num_layers]
        self.logger.info("Distance encoding: {}".format(self.enc))

    def init_internal_data(self, src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt):
        if self.enc_dim == 0:
            return
        start = time.time()
        # initialize internal data structure to index node positions
        self.nodetime2emb_maps = self.collect_pos_mapping_ptree(src_idx_l, tgt_idx_l, cut_time_l, subgraph_src,
                                                                subgraph_tgt)
        end = time.time()
        if self.verbosity > 1:
            self.logger.info('init positions encodings for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))

    def collect_pos_mapping_ptree(self, src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt):
        # Return:
        # nodetime2idx_maps: a list of dict {(node index, rounded time string) -> index in embedding look up matrix}
        if self.cpu_cores == 1:
            subgraph_src_node, _, subgraph_src_ts = subgraph_src  # only use node index and timestamp to identify a node in temporal graph
            subgraph_tgt_node, _, subgraph_tgt_ts = subgraph_tgt
            nodetime2emb_maps = {}
            for row in range(len(src_idx_l)):
                src = src_idx_l[row]
                tgt = tgt_idx_l[row]
                cut_time = cut_time_l[row]
                src_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node]
                src_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts]
                tgt_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_node]
                tgt_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_ts]
                nodetime2emb_map = PositionEncoder.collect_pos_mapping_ptree_sample(src, tgt, cut_time,
                                                                   src_neighbors_node, src_neighbors_ts,
                                                                   tgt_neighbors_node, tgt_neighbors_ts, batch_idx=row, enc=self.enc)
                nodetime2emb_maps.update(nodetime2emb_map)
        else:
            # multiprocessing version, no significant gain though
            cores = self.cpu_cores
            if cores in [-1, 0]:
                cores = mp.cpu_count()
            pool = mp.Pool(processes=cores)
            nodetime2emb_maps = pool.map(PositionEncoder.collect_pos_mapping_ptree_sample_mp,
                                         [(src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt, row, self.enc) for row in range(len(src_idx_l))],
                                         chunksize=len(src_idx_l)//cores+1)
            pool.close()
        return nodetime2emb_maps

    @staticmethod
    def collect_pos_mapping_ptree_sample(src, tgt, cut_time, src_neighbors_node, src_neighbors_ts,
                                         tgt_neighbors_node, tgt_neighbors_ts, batch_idx, enc='spd'):
        """
        This function has the potential of being written in numba by using numba.typed.Dict!
        """
        n_hop = len(src_neighbors_node)
        makekey = nodets2key
        nodetime2emb = {}
        if enc == 'spd':
            for k in range(n_hop-1, -1, -1):
                for src_node, src_ts, tgt_node, tgt_ts in zip(src_neighbors_node[k], src_neighbors_ts[k],
                                                              tgt_neighbors_node[k], tgt_neighbors_ts[k]):

                    src_key, tgt_key = makekey(batch_idx, src_node, src_ts), makekey(batch_idx, tgt_node, tgt_ts)
                    # src_ts, tgt_ts = PositionEncoder.float2str(src_ts), PositionEncoder.float2str(tgt_ts)
                    # src_key, tgt_key = (src_node, src_ts), (tgt_node, tgt_ts)
                    if src_key not in nodetime2emb:
                        nodetime2emb[src_key] = [k+1, 2*n_hop]  # 2*n_hop for disconnected case
                    else:
                        nodetime2emb[src_key][0] = k+1
                    if tgt_key not in nodetime2emb:
                        nodetime2emb[tgt_key] = [2*n_hop, k+1]
                    else:
                        nodetime2emb[tgt_key][1] = k+1
            # add two end nodes
            src_key = makekey(batch_idx, src, cut_time)
            tgt_key = makekey(batch_idx, tgt, cut_time)
            # src_key = (src, PositionEncoder.float2str(cut_time))
            # tgt_key = (tgt, PositionEncoder.float2str(cut_time))
            if src_key in nodetime2emb:
                nodetime2emb[src_key][0] = 0
            else:
                nodetime2emb[src_key] = [0, 2*n_hop]
            if tgt_key in nodetime2emb:
                nodetime2emb[tgt_key][1] = 0
            else:
                nodetime2emb[tgt_key] = [2*n_hop, 0]
            null_key = makekey(batch_idx, 0, 0.0)
            nodetime2emb[null_key] = [2 * n_hop, 2 * n_hop]
            # nodetime2emb[(0, PositionEncoder.float2str(0.0))] = [2*n_hop, 2*n_hop] # Fix a big bug with 0.0! Also, very important to keep null node far away from the two end nodes!
        elif enc == 'lp':
            # landing probability encoding, n_hop+1 types of probabilities for each node
            src_neighbors_node, src_neighbors_ts = [[src]] + src_neighbors_node, [[cut_time]] + src_neighbors_ts
            tgt_neighbors_node, tgt_neighbors_ts = [[tgt]] + tgt_neighbors_node, [[cut_time]] + tgt_neighbors_ts
            for k in range(n_hop+1):
                k_hop_total = len(src_neighbors_node[k])
                for src_node, src_ts, tgt_node, tgt_ts in zip(src_neighbors_node[k], src_neighbors_ts[k],
                                                              tgt_neighbors_node[k], tgt_neighbors_ts[k]):
                    src_key, tgt_key = makekey(batch_idx, src_node, src_ts), makekey(batch_idx, tgt_node, tgt_ts)
                    # src_ts, tgt_ts = PositionEncoder.float2str(src_ts), PositionEncoder.float2str(tgt_ts)
                    # src_key, tgt_key = (src_node, src_ts), (tgt_node, tgt_ts)
                    if src_key not in nodetime2emb:
                        nodetime2emb[src_key] = np.zeros((2, n_hop+1), dtype=np.float32)
                    if tgt_key not in nodetime2emb:
                        nodetime2emb[tgt_key] = np.zeros((2, n_hop+1), dtype=np.float32)
                    nodetime2emb[src_key][0, k] += 1/k_hop_total  # convert into landing probabilities by normalizing with k hop sampling number
                    nodetime2emb[tgt_key][1, k] += 1/k_hop_total  # convert into landing probabilities by normalizing with k hop sampling number
            null_key = makekey(batch_idx, 0, 0.0)
            nodetime2emb[null_key] = np.zeros((2, n_hop + 1), dtype=np.float32)
            # nodetime2emb[(0, PositionEncoder.float2str(0.0))] = np.zeros((2, n_hop+1), dtype=np.float32)
        else:
            assert(enc == 'saw')  # self-based anonymous walk, no mutual distance encoding
            src_neighbors_node, src_neighbors_ts = [[src]] + src_neighbors_node, [[cut_time]] + src_neighbors_ts
            tgt_neighbors_node, tgt_neighbors_ts = [[tgt]] + tgt_neighbors_node, [[cut_time]] + tgt_neighbors_ts
            src_seen_nodes2label = {}
            tgt_seen_nodes2label = {}
            for k in range(n_hop + 1):
                for src_node, src_ts, tgt_node, tgt_ts in zip(src_neighbors_node[k], src_neighbors_ts[k],
                                                              tgt_neighbors_node[k], tgt_neighbors_ts[k]):
                    src_key, tgt_key = makekey(batch_idx, src_node, src_ts), makekey(batch_idx, tgt_node, tgt_ts)
                    # src_ts, tgt_ts = PositionEncoder.float2str(src_ts), PositionEncoder.float2str(tgt_ts)
                    # src_key, tgt_key = (src_node, src_ts), (tgt_node, tgt_ts)

                    # encode src node tree
                    if src_key not in nodetime2emb:
                        nodetime2emb[src_key] = np.zeros((n_hop + 1, ), dtype=np.float32)
                    if src_node not in src_seen_nodes2label:
                        new_src_node_label = k
                        src_seen_nodes2label[src_key] = k
                    else:
                        new_src_node_label = src_seen_nodes2label[src_node]
                    nodetime2emb[src_key][new_src_node_label] = 1

                    # encode tgt node tree
                    if tgt_key not in nodetime2emb:
                        nodetime2emb[tgt_key] = np.zeros((n_hop + 1, ), dtype=np.float32)
                    if tgt_node not in tgt_seen_nodes2label:
                        new_tgt_node_label = k
                        tgt_seen_nodes2label[tgt_node] = k
                    else:
                        new_tgt_node_label = tgt_seen_nodes2label[tgt_node]
                    nodetime2emb[src_key][new_tgt_node_label] = 1
            null_key = makekey(batch_idx, 0, 0.0)
            nodetime2emb[null_key] = np.zeros((n_hop + 1, ), dtype=np.float32)
            # nodetime2emb[(0, PositionEncoder.float2str(0.0))] = np.zeros((n_hop + 1, ), dtype=np.float32)
        # for key, value in nodetime2emb.items():
        #     nodetime2emb[key] = torch.tensor(value)
        return nodetime2emb

    def forward(self, node_record, t_record):
        '''
        accept two numpy arrays each of shape [batch, k-hop-support-number], corresponding to node indices and timestamps respectively
        return Torch.tensor: position features of shape [batch, k-hop-support-number, position_dim]
        return Torch.tensor: position features of shape [batch, k-hop-support-number, position_dim]
        '''
        # encodings = []
        device = next(self.projection.parameters()).device
        # float2str = PositionEncoder.float2str
        batched_keys = make_batched_keys(node_record, t_record)
        unique, inv = np.unique(batched_keys, return_inverse=True)
        unordered_encodings = np.array([self.nodetime2emb_maps[key] for key in unique])
        encodings = unordered_encodings[inv, :]
        encodings = torch.tensor(encodings).to(device)
        walk_encodings = None
        # walk_encodings = encodings.view(encodings.shape[0], -1, encodings.shape[-1], *encodings.shape[-2:]) # this line of code is current bugged
        # for batch_idx, (n_l, ts_l) in enumerate(zip(node_record, t_record)):
        #     # encoding = [self.nodetime2emb_maps[batch_idx][(n, float2str(ts))] for n, ts in zip(n_l, ts_l)]
        #     # encodings.append(torch.stack(encoding))  # shape [support_n, 2] / [support_n, 2, num_layers+1]
        #     lookup_func = np.vectorize(self.nodetime2emb_maps[batch_idx].get)
        #     encodings = lookup_func(np.array(zip(node_record, [float2str(ts) for ts in t_record])))
        # encodings = torch.stack(encodings).to(device)  # shape [B, support_n, 2] / [B, support_n, 2, num_layers+1]
        common_nodes = (((encodings.sum(-1) > 0).sum(-1) == 2).sum().float() / (encodings.shape[0] * encodings.shape[1])).item()
        encodings = self.get_trainable_encodings(encodings)
        return encodings, common_nodes, walk_encodings

    @staticmethod
    def collect_pos_mapping_ptree_sample_mp(args):
        src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt, row, enc = args
        subgraph_src_node, _, subgraph_src_ts = subgraph_src  # only use node index and timestamp to identify a node in temporal graph
        subgraph_tgt_node, _, subgraph_tgt_ts = subgraph_tgt
        src = src_idx_l[row]
        tgt = tgt_idx_l[row]
        cut_time = cut_time_l[row]
        src_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node]
        src_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts]
        tgt_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_node]
        tgt_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_ts]
        nodetime2emb_map = PositionEncoder.collect_pos_mapping_ptree_sample(src, tgt, cut_time,
                                                                            src_neighbors_node, src_neighbors_ts,
                                                                            tgt_neighbors_node, tgt_neighbors_ts, enc=enc)
        return nodetime2emb_map

    def get_trainable_encodings(self, encodings):
        '''
        Args:
            encodings: a device tensor of shape [batch, support_n, 2] / [batch, support_n, 2, L+1]
        Returns:  a device tensor of shape [batch, pos_dim]
        '''
        if self.enc == 'spd':
            encodings[encodings > (self.num_layers+0.5)] = self.num_layers + 1
            encodings = self.trainable_embedding(encodings.long())  # now shape [batch, support_n, 2, pos_dim]
            encodings = encodings.sum(dim=-2)  # now shape [batch, support_n, pos_dim]
        elif self.enc == 'lp':
            encodings = self.trainable_embedding(encodings.float())   # now shape [batch, support_n, 2, pos_dim]
            encodings = encodings.sum(dim=-2)  # now shape [batch, support_n, pos_dim]
        else:
            assert(self.enc == 'saw')
            encodings = self.trainable_embedding(encodings.float())  # now shape [batch, support_n, pos_dim]
        return encodings


class FeatureEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, dropout_p=0.1):
        super(FeatureEncoder, self).__init__()
        self.hidden_features_one_direction = hidden_features//2
        self.model_dim = self.hidden_features_one_direction * 2  # notice that we are using bi-lstm
        if self.model_dim == 0:  # meaning that this encoder will be use less
            return
        self.lstm_encoder = nn.LSTM(input_size=in_features, hidden_size=self.hidden_features_one_direction, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X, mask=None):
        batch, n_walk, len_walk, feat_dim = X.shape
        X = X.view(batch*n_walk, len_walk, feat_dim)
        if mask is not None:
            lengths = mask.view(batch*n_walk)
            X = pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded_features = self.lstm_encoder(X)[0]
        if mask is not None:
            encoded_features, lengths = pad_packed_sequence(encoded_features, batch_first=True)
        encoded_features = encoded_features.select(dim=1, index=-1).view(batch, n_walk, self.model_dim)
        encoded_features = self.dropout(encoded_features)
        return encoded_features