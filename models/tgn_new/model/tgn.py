import numpy as np
import torch
from collections import defaultdict
import time


from ..utils import MergeLayer, process_sampling_numbers
from ..modules.memory import Memory
from ..modules.message_aggregator import get_message_aggregator
from ..modules.message_function import get_message_function
from ..modules.memory_updater import get_memory_updater
from ..modules.embedding_module import get_embedding_module
from ..modules.caw_encoder import PositionEncoder, FeatureEncoder

from .time_encoding import TimeEncode
from log_utils import get_logger


class TGN(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
                 n_heads=2, dropout=0.1, use_memory=False,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=500, embedding_module_type="graph_attention",
                 message_function="mlp",
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
                 memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False,
                 dyrep=False,
                 pos_dim=100, pos_enc='spd', caw_layers=2, caw_neighbors=[64, 2], caw_dropout=0.1, use_caw_lstm=True,
                 use_caw=True, use_caw_embed=True, use_caw_message=True):
        super(TGN, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = get_logger(__name__)

        self.node_raw_features = node_features
        self.edge_raw_features = edge_features

        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep

        # TODO: CAW-specific parameters here
        self.use_caw = use_caw
        caw_neighbors[1] = caw_layers #TODO: this is a temporary workaround
        self.use_caw_message = use_caw_message
        self.use_caw_embed = use_caw_embed
        self.caw_feat_dim = 0
        self.use_caw_lstm = use_caw_lstm
        if self.use_caw:
            self.caw_ngh_finder = None
            self.pos_dim = pos_dim  # position feature dimension
            self.pos_enc = pos_enc

            if self.use_caw_lstm:
                self.caw_feat_dim = self.pos_dim
            else:
                self.caw_feat_dim = self.pos_dim * (caw_layers + 1)
            self.caw_neighbors, self.caw_layers = process_sampling_numbers(
                caw_neighbors, caw_layers)
            self.position_encoder = PositionEncoder(enc_dim=self.pos_dim, num_layers=self.caw_layers,
                                                    ngh_finder=self.caw_ngh_finder,
                                                    logger=self.logger,
                                                    enc=self.pos_enc)
            self.caw_dropout = caw_dropout
            self.use_caw_lstm = use_caw_lstm
            if self.use_caw_lstm:
                self.walk_lstm_encoder = FeatureEncoder(
                    self.pos_dim, self.pos_dim, self.caw_dropout)

        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.time_encoder = self.time_encoder.to(device)
        self.memory = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + self.time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
            if self.use_caw_message:
                message_dimension += self.caw_feat_dim
            self.memory = Memory(n_nodes=self.n_nodes,
                                 memory_dimension=self.memory_dimension,
                                 input_dimension=message_dimension,
                                 message_dimension=message_dimension,
                                 device=device)
            self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                             device=device)
            self.message_function = get_message_function(module_type=message_function,
                                                         raw_message_dimension=raw_message_dimension,
                                                         message_dimension=message_dimension)
            self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                     memory=self.memory,
                                                     message_dimension=message_dimension,
                                                     memory_dimension=self.memory_dimension,
                                                     device=device)

        self.embedding_module_type = embedding_module_type

        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     node_features=self.node_raw_features,
                                                     edge_features=self.edge_raw_features,
                                                     memory=self.memory,
                                                     neighbor_finder=self.neighbor_finder,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.n_node_features,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=self.n_node_features,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     use_memory=use_memory,
                                                     n_neighbors=self.n_neighbors,
                                                     caw_feat_dim=self.caw_feat_dim*int(self.use_caw_embed))

        # MLP to compute probability on an edge given two node embeddings
        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                         self.n_node_features,
                                         1).to(self.device)

    def update_caw_ngh_finder(self, ngh_finder):
        self.caw_ngh_finder = ngh_finder
        self.position_encoder.ngh_finder = ngh_finder

    def grab_subgraph(self, src_idx_l, cut_time_l, e_idx_l=None):
        # TODO: TGN ngh finder differs from that of CAW; CAW implementation used here
        subgraph = self.caw_ngh_finder.find_k_hop(self.caw_layers, src_idx_l, cut_time_l, num_neighbors=self.caw_neighbors,
                                                  e_idx_l=e_idx_l)
        return subgraph

    def subgraph_tree2walk(self, src_idx_l, cut_time_l, subgraph_src):
        # put src nodes and extracted subgraph together
        node_records, eidx_records, t_records = subgraph_src
        node_records_tmp = [np.expand_dims(src_idx_l, 1)] + node_records
        eidx_records_tmp = [np.zeros_like(node_records_tmp[0])] + eidx_records
        t_records_tmp = [np.expand_dims(cut_time_l, 1)] + t_records

        # use the list to construct a new matrix
        new_node_records = self.subgraph_tree2walk_one_component(
            node_records_tmp)
        new_eidx_records = self.subgraph_tree2walk_one_component(
            eidx_records_tmp)
        new_t_records = self.subgraph_tree2walk_one_component(t_records_tmp)
        return new_node_records, new_eidx_records, new_t_records

    def subgraph_tree2walk_one_component(self, record_list):
        batch, n_walks, walk_len, dtype = record_list[0].shape[0], record_list[-1].shape[-1], len(
            record_list), record_list[0].dtype
        record_matrix = np.empty((batch, n_walks, walk_len), dtype=dtype)
        for hop_idx, hop_record in enumerate(record_list):
            assert(n_walks % hop_record.shape[-1] == 0)
            record_matrix[:, :, hop_idx] = np.repeat(
                hop_record, repeats=n_walks // hop_record.shape[-1], axis=1)
        return record_matrix

    def caw_retrieve_position_features(self, src_idx_l, node_records, cut_time_l, t_records):
        start = time.time()
        encode = self.position_encoder
        if encode.enc_dim == 0:
            return None
        batch, n_walk, len_walk = node_records.shape
        node_records_r, t_records_r = node_records.reshape(
            batch, -1), t_records.reshape(batch, -1)
        # if test:
        #     self.walk_encodings_scores['encodings'].append(walk_encodings)
        position_features, common_nodes, walk_encodings = encode(
            node_records_r, t_records_r)
        position_features = position_features.view(
            batch, n_walk, len_walk, self.pos_dim)
        # TODO: kicking this for now(only this line): self.update_common_node_percentages(common_nodes)
        # if test:
        #     self.walk_encodings_scores['encodings'].append(walk_encodings)
        end = time.time()
        # if self.verbosity > 1:
        #    self.logger.info('encode positions encodings for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
        return position_features

    def caw_compute_edge_embeddings(self, source_nodes, destination_nodes, edge_times, subgraphs):
        src_idx_l = source_nodes
        tgt_idx_l = destination_nodes
        cut_time_l = edge_times
        subgraph_src, subgraph_tgt = subgraphs
        self.position_encoder.init_internal_data(
            src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt)
        subgraph_src = self.subgraph_tree2walk(
            src_idx_l, cut_time_l, subgraph_src)
        subgraph_tgt = self.subgraph_tree2walk(tgt_idx_l, cut_time_l, subgraph_tgt)
        node_records_src, eidx_records_src, t_records_src = subgraph_src
        position_features_src = self.caw_retrieve_position_features(src_idx_l, node_records_src, cut_time_l,
                                                                    t_records_src)
        node_records_tgt, eidx_records_tgt, t_records_tgt = subgraph_tgt
        position_features_tgt = self.caw_retrieve_position_features(tgt_idx_l, node_records_tgt, cut_time_l, t_records_tgt)

        # TODO: adding Bi-LSTM encoding of each walk here
        if self.use_caw_lstm:
            position_features_src = self.walk_lstm_encoder(position_features_src)
            position_features_tgt = self.walk_lstm_encoder(position_features_tgt)
        #node_records_tgt, eidx_records_tgt, t_records_tgt = subgraph_tgt
        # position_features_tgt = self.caw_retrieve_position_features(tgt_idx_l, node_records_tgt, cut_time_l,
        #                                                            t_records_tgt)
        return [position_features_src, position_features_tgt]

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                    edge_idxs, n_neighbors=20):
        """
        Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

        source_nodes [batch_size]: source ids.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Temporal embeddings for sources, destinations and negatives
        """

        n_samples = len(source_nodes)
        nodes = np.concatenate(
            [source_nodes, destination_nodes, negative_nodes]).astype(int)
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])

        memory = None
        time_diffs = None

        # TODO: CAW position vectors computed here; pass them to the message function; also, dont forget to change message dimesions
        if self.use_caw_message:
            subgraphs = [self.grab_subgraph(source_nodes, edge_times, edge_idxs), self.grab_subgraph(
                destination_nodes, edge_times, edge_idxs)]
            caw_features = self.caw_compute_edge_embeddings(
                source_nodes, destination_nodes, edge_times, subgraphs)
        else:
            caw_features = None

        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory for all nodes with messages stored in previous batches
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                              self.memory.messages)
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

      # Compute differences between the time the memory of a node was last updated,
      # and the time for which we want to compute the embedding of a node
            source_time_diffs = torch.LongTensor(edge_times).to(
                self.device) - last_update[source_nodes].long()
            source_time_diffs = (
                source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
            destination_time_diffs = torch.LongTensor(edge_times).to(
                self.device) - last_update[destination_nodes].long()
            destination_time_diffs = (
                destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
            negative_time_diffs = torch.LongTensor(edge_times).to(
                self.device) - last_update[negative_nodes].long()
            negative_time_diffs = (
                negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

            time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                                   dim=0)

        # Compute the embeddings using the embedding module
        if self.use_caw_embed:
            # TODO: add CAW embedding use option
            node_ngh_ids = self.embedding_module.get_ngh_ids(memory=memory,
                                                                    source_nodes=nodes,
                                                                    timestamps=timestamps,
                                                                    n_layers=self.n_layers,
                                                                    n_neighbors=n_neighbors,
                                                                    time_diffs=time_diffs)
            nodes_latent = np.repeat(nodes, n_neighbors)
            times_latent = np.repeat(edge_times.min(), nodes.shape[0] * n_neighbors)
            node_ngh_ids = node_ngh_ids.reshape(-1)
            subgraphs = [self.grab_subgraph(nodes_latent, times_latent), self.grab_subgraph(node_ngh_ids, times_latent)]
            caw_features_emb = self.caw_compute_edge_embeddings(nodes_latent, node_ngh_ids, times_latent, subgraphs)
            caw_features_emb = caw_features_emb[0] + caw_features_emb[1]
            #caw_features_emb = caw_features_emb.reshape(nodes.shape[0], n_neighbors, -1, self.caw_neighbors[0] * 2).mean(axis=3)
            caw_features_emb = caw_features_emb.reshape(nodes.shape[0], n_neighbors, self.caw_feat_dim, -1).mean(axis=3)
            caw_features_emb = caw_features_emb.to(self.device)
            node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                source_nodes=nodes,
                                                                timestamps=timestamps,
                                                                n_layers=self.n_layers,
                                                                n_neighbors=n_neighbors,
                                                                time_diffs=time_diffs,
                                                                caw_features=caw_features_emb)
        else:
            node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                    source_nodes=nodes,
                                                                    timestamps=timestamps,
                                                                    n_layers=self.n_layers,
                                                                    n_neighbors=n_neighbors,
                                                                    time_diffs=time_diffs)

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples:]

        if self.use_memory:
            if self.memory_update_at_start:
                # Persist the updates to the memory only for sources and destinations (since now we have
                # new messages for them)
                self.update_memory(positives, self.memory.messages)

                assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
                    "Something wrong in how the memory was updated"

                # Remove messages for the positives since we have already updated the memory using them
                self.memory.clear_messages(positives)

        unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                      source_node_embedding,
                                                                      destination_nodes,
                                                                      destination_node_embedding,
                                                                      edge_times, edge_idxs, caw_features)
        unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                                destination_node_embedding,
                                                                                source_nodes,
                                                                                source_node_embedding,
                                                                                edge_times, edge_idxs, caw_features)
        if self.memory_update_at_start:
            self.memory.store_raw_messages(
                unique_sources, source_id_to_messages)
            self.memory.store_raw_messages(
                unique_destinations, destination_id_to_messages)
        else:
            self.update_memory(unique_sources, source_id_to_messages)
            self.update_memory(unique_destinations, destination_id_to_messages)

        if self.dyrep:
            source_node_embedding = memory[source_nodes]
            destination_node_embedding = memory[destination_nodes]
            negative_node_embedding = memory[negative_nodes]

        return source_node_embedding, destination_node_embedding, negative_node_embedding

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                   edge_idxs, n_neighbors=20):
        """
        Compute probabilities for edges between sources and destination and between sources and
        negatives by first computing temporal embeddings using the TGN encoder and then feeding them
        into the MLP decoder.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Probabilities for both the positive and negative edges
        """
        n_samples = len(source_nodes)
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

        score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                    torch.cat([destination_node_embedding,
                                               negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score.sigmoid(), neg_score.sigmoid()

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(
                unique_messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)

    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(
                unique_messages)

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)

        return updated_memory, updated_last_update

    def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                         destination_node_embedding, edge_times, edge_idxs, caw_features):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_raw_features[edge_idxs].float().to(
            self.device)
        # edge_features = torch.from_numpy(self.edge_raw_features[edge_idxs]).float().to(self.device)

        source_memory = self.memory.get_memory(source_nodes) if not \
            self.use_source_embedding_in_message else source_node_embedding
        destination_memory = self.memory.get_memory(destination_nodes) if \
            not self.use_destination_embedding_in_message else destination_node_embedding

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(
            source_time_delta.unsqueeze(dim=1)).view(len(source_nodes), -1)

        # TODO: concatenating CAW positional embeddings here
        if self.use_caw_message:
            if self.use_caw_lstm:
                caw_features = caw_features[0].mean(axis=1)
            else:
                caw_features = caw_features[0].mean(axis=1).reshape(caw_features[0].shape[0], -1)
            #+ caw_features[1].mean(axis=1).reshape(caw_features[1].shape[0], -1)
            #caw_features = torch.from_numpy(caw_features)
            caw_features = caw_features.to(self.device)
            #assert False
            source_message = torch.cat([source_memory, destination_memory, edge_features,
                                    source_time_delta_encoding, caw_features],
                                    dim=1)
        else:
            source_message = torch.cat([source_memory, destination_memory, edge_features,
                                        source_time_delta_encoding],
                                    dim=1)
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append(
                (source_message[i], edge_times[i]))

        return unique_sources, messages

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
