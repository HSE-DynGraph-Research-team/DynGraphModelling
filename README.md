# DynGraph-modelling

Repository for dynamic graph modelling experiments. 

Abstract interfaces for graph models and data are implemented here to allow for more wide exploration of dynamic graph topics.


## Contents

Model interface is located at `model_wrapper.py` and provides list of methods with respective signatures to be implemented for proper model training and evaluation. An example of such implementation is provided in `tgn_wrapper.py`.

Data interface is located at `data_interface.py` and implements reading transaction- and snapshot-based temporal graph data. Loaded data is able to be batched in either transaction or snapshot fashion.

[More info on how data processing is organized.](data_model_guide.md)

General pipelines are located at `pipelines.py` and implements training and evaluation of the models.
Every step of the pipeline (training, various inference) is given up for model wrapper to be implemented; pipeline implements general sequence of actions and training setting.

Evaluation routines are provided in `evaluation.py` and provide functions for evaluating two general tasks - binary node classification and temporal edge prediction. Both of these tasks can be run in either transductive (predicting feature of an already seen object) and inductive (predicting feature of an unseen object) settings; resulting 4 tasks are being evaluated in `pipelines.py`.

## Current issues and tasks
- ~~finish batch dispension routines for data interface~~
- ~~add splitting options for train-test-val in data interface~~
- ~~integrate data interface into current models~~
- ~~add scenarios for batching and training~~
- ~~add raw transaction options in data interface~~
- add raw graph reading options in data interface
- add more models via new interfaces (PyGT to start with)
- more informative and systematic approach to metric reporting


## Launching model

To launch a model traing evaluation (tested in python v.3.8):
1. Install requirements (`pip install -r requirements.txt`)
2. Download wikipedia data from [here](http://snap.stanford.edu/jodie/#datasets) and put the dataset into `data` subfolder
3. 
    - Run `python pipelines.py` for testing single pipeline 
    - Run `python scenarios.py` for testing full scenatrio
