import numpy as np

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels, seed=None):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)

    def get_batch(self, s, e):
        sources_batch      = self.sources[s:e]
        destinations_batch = self.destinations[s:e]
        edge_idxs_batch    = self.edge_idxs[s:e]
        timestamps_batch   = self.timestamps[s:e]
        labels_batch       = self.labels[s:e]
        return sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, labels_batch
        

