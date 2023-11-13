# README

## Overview

This package contains code modified from the paper "TransTab: A Flexible Transferable Tabular Learning Framework." The original project, available at [TransTab GitHub Repository](https://github.com/RyanWangZf/transtab), is under the BSD 2-Clause license. The code here has been modified to focus specifically on numerical features, retaining only methods relevant to these features. The training approach is limited to unsupervised training. Differing from the original paper's usage of TransTab for final predictions, this code is utilized for feature extraction.

## Contents

## Handling heterogeneous learnwares

The code is used for finding a unified specification space for learnwares generated from table data with heterogeneous feature spaces and assigning new specifications accordingly. When the market receives some leanrwares, it utilize existing learnware specifications to train an engine. This engine integrates the specifications from various spaces into a unified "specification world", assigning new market-specific specifications to the learnware. As more learnwares are uploaded, the engine continuously updates, refining the specification world and updating the specifications of the learnware.

## License

The hetero_map package, based on the TransTab project, adheres to the BSD 2-Clause license.