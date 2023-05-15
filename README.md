# A Combinatorial Multi-Armed Bandit Approach to Correlation Clustering

## Overview

This project is developed as part of the following research paper:

F. Gullo, D. Mandaglio, A. Tagarelli (2023). *A Combinatorial Multi-Armed Bandit Approach to Correlation Clustering* published in Data Mining and Knowledge Discovery (DAMI), 2023

Please cite the above paper in any research publication you may produce using this code or data/analysis derived from it.


## Folders
- datasets:  contains the original data as well as the preprocessed data (as described in the paper). Biggest networks (to be unzipped in datasets folder) can be downloaded at the following [link](https://drive.google.com/open?id=1r0krGQMm0QyUAbJlGZSxTmm2ULaajBKI)
- code: it contains this project code
- output: it stores all results produced by the algorithms 

## Usage

From the folder 'CMAB-CC/code', run the following command:
```bash      

run_CMAB.py [-h] -d DATASET_NAME [-b {cc-clcb,cc-clcb-m,global-clcb,global-clcb-m,eg,eg-fixed,pe,cts,pcexp-clcb}] [-eps EXPLORATION_PROBABILITY] [-o {pivot,charikar}] [-T TIMESTEPS]
                   [-r RUNS] [-s SEED]                            
```
### Dependencies
- numpy
- python-igraph


### Positional arguments
```
  -d DATASET, --dataset DATASET
                        Input dataset, whose name identifies a particular subfolder in 'datasets/'
```
### Optional arguments
```                                          
  -h, --help            show this help message and exit
  -b {cc-clcb,cc-clcb-m,global-clcb,global-clcb-m,eg,eg-fixed,pe,cts,pcexp-clcb}, --bandit {cc-clcb,cc-clcb-m,global-clcb,global-clcb-m,eg,eg-fixed,pe,cts,pcexp-clcb}
                        Bandit algorithm
  -eps EXPLORATION_PROBABILITY, --exploration_probability EXPLORATION_PROBABILITY
                        exploration probability for epsilon-greedy CMAB algorithm
  -o {pivot,charikar}, --oracle {pivot,charikar}
                        Oracle to use in each CMAB step
  -T TIMESTEPS, --timesteps TIMESTEPS
                        Number of timesteps to run the selected bandit algorithm
  -r RUNS, --runs RUNS  Number of bandit runs
  -s SEED, --seed SEED  Random generation seed -- for reproducibility (default value 100)            
                    
```

