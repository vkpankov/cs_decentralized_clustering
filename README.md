# README

## Overview

Code for the Paper: Communication-Efficient Decentralized Clustering in Dynamical Multi-Agent Systems

This repository implements a decentralized clustering algorithm using compressed sensing for communication efficiency and neural network-based centroids (set) prediction from compressed data  representation.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cs_decentralized_clustering
   cd cs_decentralized_clustering
   ```

2. Install the required dependencies (torch):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

The `train.py` script trains the reconstruction model using synthetic dataset (dataset.py). Below is an example of how to run the script:

```bash
python train.py \
    --clusters_num 10 \
    --agents_num 1000 \
    --grid_size 128 \
    --comp_size 12 12 \
    --batch_size 128 \
    --epochs 5 \
    --lr 0.001 \
    --checkpoint_dir checkpoints
```

### Inference example

python lvp_simulation.py \
    --grid_size 128 \
    --agents_num 1000 \
    --max_clusters 10 \
    --batch_size 32 \
    --comp_size 12 12 \
    --step_size 0.2 \
    --num_it_list 10 15 20 25 30 35 40 45 50 100 \
    --num_simulations 30 \
    --graph_type "random" \
    --model_ckpt "/Users/vikentiy/Documents/cs_decentralized_clustering/checkpoints/last_c10_p12x12.pt" \
    --pickle_path "results_12x12.pkl"

### Arguments

| Argument          | Description                                                | Default       |
|-------------------|------------------------------------------------------------|---------------|
| `--clusters_num`  | Number of clusters.                                         | `10`          |
| `--agents_num`    | Number of agents.                                           | `1000`        |
| `--grid_size`     | Grid size for mapping agent states.                        | `128`         |
| `--comp_size`     | Compressed grid size (height, width).                      | `(12, 12)`    |
| `--batch_size`    | Batch size for training.                                   | `128`         |
| `--epochs`        | Number of training epochs.                                 | `5`           |
| `--lr`            | Initial learning rate for the optimizer.                          | `0.001`       |
| `--checkpoint_dir`| Directory to save model checkpoints.                      | `checkpoints` |
