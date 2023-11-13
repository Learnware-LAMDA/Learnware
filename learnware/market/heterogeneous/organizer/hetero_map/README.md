# README

## Overview

This package contains code modified from the paper "TransTab: A Flexible Transferable Tabular Learning Framework." The original project, available at [TransTab GitHub Repository](https://github.com/RyanWangZf/transtab), is under the BSD 2-Clause license. The code here has been modified to focus specifically on numerical features, retaining only methods relevant to these features. The training approach is limited to unsupervised training. Differing from the original paper's usage of TransTab for final predictions, this code is utilized for feature extraction.

## Contents

- `__init__.py`: The `__init__.py` file defines the `HeteroMap` class, which forms the main network structure of the market engine. It includes methods for handling heterogeneous tabular data, focusing on mapping data from diverse feature spaces into a unified "specification world".
- `trainer.py`: The `trainer.py` file focuses on the unsupervised training process of the market engine. The `TransTabCollatorForCL` class is used for generating positive and negative samples from tabular vertical partitions for unsupervised learning.
- `feature_extractor.py`: This file is utilized for the purpose of tokenizing feature descriptions and transforming them into word embeddings.

## Handling heterogeneous learnwares

The code is used for finding a unified specification space for learnwares generated from table data with heterogeneous feature spaces and assigning new specifications accordingly. When the market receives some leanrwares, it utilize existing learnware specifications to train an engine. This engine integrates the specifications from various spaces into a unified "specification world", assigning new market-specific specifications to the learnware. As more learnwares are uploaded, the engine continuously updates, refining the specification world and updating the specifications of the learnware.

## License

The hetero_map package, based on the TransTab project, adheres to the BSD 2-Clause license.