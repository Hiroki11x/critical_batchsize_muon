# Analysis of Muon's Convergence and Critical Batch Size

This repository contains the source code for the experiments and analysis presented in the paper "Analysis of Muon's Convergence and Critical Batch Size".

## Abstract

This paper presents a theoretical analysis of Muon, a new optimizer that respects the inherent matrix structure of neural network parameters. We give convergence proofs for four practical variants of Muon: with Nesterov momentum, without Nesterov momentum, with weight decay and without weight decay. We then show that adding weight decay leads to strictly tighter bounds on both the norm of the parameters and the norm of the gradients, and we describe the relationship between the weight decay coefficient and the learning rate. Finally, we determine the optimal batch size for Muon and validate our theoretical findings with experiments.

## Repository Structure

- `src/`: Contains the Python source code for the Muon optimizer, models, and training scripts.
- `notebook/`: Jupyter notebooks for analyzing results and generating plots for the paper.
- `slurm_cluster/`: Shell scripts for running experiments on different high-performance computing clusters.

## Usage

### Prerequisites
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Running Experiments
The experiment scripts are located in the respective cluster directories (e.g., `slurm_cluster/`). These scripts are configured to run the training jobs on the cluster's queuing system.

For example, to run an experiment on the Slurm cluster:
```bash
cd slurm_cluster/cbs_1gpu/
sbatch run_muon_with_nesterov_exp1.sh
```

### Reproducing Figures
The Jupyter notebooks in the `notebook/` directory can be used to reproduce the figures and analysis from the paper.


