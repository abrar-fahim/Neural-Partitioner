# Neural-Partitioner

This repository contains code for the paper labelled "Unsupervised Space Partitioning for Nearest Neighbor Search" by Abrar Fahim, [Mohammed Eunus Ali](https://sites.google.com/site/mohammedeunusali/), [Muhammad Aamir Cheema](http://www.aamircheema.com/).

## Code Organization

The entry point of our code is [main.py](main.py). [main.py](main.py) uses the [paths.txt](paths.txt) file to locate the datasets to load for training. 

## Configuring the [paths.txt](paths.txt) file
This file contains all the directory paths that our code needs for various tasks. All the paths listed here must be absolute paths.
- `paths_to_mnist`: path to the folder containing the MNIST dataset in `hdf5` format.
- `path_to_sift`:  similar to `path_to_mnist` for the SIFT dataset.
- `path_to_knn_matrix`: path to folder that will store the generated k-NN matrix of the dataset.
- `path_to_models`: path to folder that will store the trained models.
- `path_to_tensors`: path to folder that will cache some of the processed tensors for faster subsequent runs.

##  Running our code
First populate the [paths.txt](paths.txt) file with the proper folder directories as outlined above. Then download the [SIFT](http://corpus-texmex.irisa.fr/) and/or [MNIST](http://yann.lecun.com/exdb/mnist/) datasets from [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks#data-sets) into the `path_to_mnist` and/or `path_to_sift` folders.

To run our code with in the default configuration, run:

 `python main.py`. 

Example of running with a custom configuration: 

`python main.py --n_bins 256 --dataset_name mnist --n_trees 1 --load_knn`

## [main.py](main.py) parameters:
Default values of the parameters are specified in [utils.py](utils.py).
### Partitioning parameters:
- `dataset_name`: the dataset to partition, `mnist` or `sift`.
- `n_bins`: number of bins to partition the dataset into.
- `k_train`: number of neighbors to use to build the k-NN matrix.
- `k_test`: number of neighbors to use to test the trained model.
- `n_bins_to_search`: number of bins to search for the nearest neighbors.
### Training parameters:
- `n_epochs`: number of epochs to train the model.
- `batch_size`: batch size for training.
- `lr`: learning rate for training.
- `n_trees`: number of trees to use in ensemble.
- `n_levels`: number of levels in each tree of ensemble.
- `tree_branching`: number of children per node in a tree.
- `model_type`: type of model to use for training, `neural` or `linear`.
### Cache parameters:
- `load_knn`: whether to load the k-NN matrix from file.
- `continue_train`: whether to continue training the model from the last checkpoint; loads models from file.


