## Internal data representation

Our approach utilizes transforming any of the ubiquitous formats of data for dyngraph tasks into unified data model, consisting of three tables:

- node_table:
    - node_id - int
    - last_updated - int or float
    - features - collection of floats
- edge table:
    - edge_id (unique and ordered) - int
    - timestamp - int or float
    - s_node_id - int
    - d_node_id - int
    - edge_features - collection of floats
- labels:
    - node_id - int
    - timestamp - int or float
    - state - int

Timesteps may not be unique for each row in any table in case of discrete time data. For compatibility's sake in case of discrete time static features in labels and edges' tables are broadcasted for each timestamp.


## Batching

Data interface supports wide variety of batching options:
- Output format:
    - Transactions (in form of TGN-compatible data)
    - Snapshots (in form of pyTorch Geometric data obj)
- Batch size: 
    - int: how many elements per batch
    - float: what fraction of total data should be in a single batch 
- Batch spec - specify every batch size (by a respective unit, see *divide_by*) in a list: 
    - list of ints: specify batch size for each batch
    - list of floats: specify fractions of each batch
    - *(**NB**: list may contain both ints and floats; excessive batches will be trimmed from the end)*
- Divide by - what should be considered a unit of batching when specifying batch size:
    - edge index: batch by unique edge ids
    - time stamp: batch by unique timestamps
    - time window: batch by time intervals

Batching options can be applied by calling the data object:
```
GC = GraphContainer()
GC.read_snapshot_data(some_data_object)
for batch in GC(batch_size=1000, divide_by='time_stamp', data_mode='transaction'):
    some_model.train(batch)
``` 



## Reading transactional data

Raw transactional data in form of CSV lacks specification on features' formats. We propose the following format:
- each row is expected to have data in the following order:
    - source node id
    - destination node id
    - timestamp
    - label
    - edge's features
    - nodes' features
- to parse features correctly it is enough to provide data on node features by including the following into dataset name:
    - node**s**=*10* if **s**ource node features consist of *10* values (no destination node features)
    - node**d**=*20* if *20* **d**estination node features are included (no source node features are provided)
    - node**b**=*30* if **b**oth destination and source nodes have *30* features
- edge feature dimensionality will be inferred from node features' dimensionality
- e.g. of a filename: `wikipedia_nodes=20.csv`




## Possible issues

- When getting batches in form of graph snapshots, features from several timesteps may be included into single batch; as a temporary solution we provide latest feature. It is recommended to make batches smaller to avoid this problem.
