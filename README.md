# A Combinatorial Multi-Armed Bandit Approach to Correlation Clustering

## Overview

This project is developed as part of the following research paper:

F. Gullo, D. Mandaglio, A. Tagarelli (2023). *A Combinatorial Multi-Armed Bandit Approach to Correlation Clustering* published in Data Mining and Knowledge Discovery (DAMI), 2023

Please cite the above paper in any research publication you may produce using this code or data/analysis derived from it.


## Folders
- datasets:  contains the original data as well as the preprocessed data (as described in the paper). Biggest networks (to be unzipped in datasets folder) can be downloaded at the following [link](https://drive.google.com/file/d/18PwfSVlNC2U5AmokNCcepmxHjLLQ4GG6/view?usp=share_link)
- code: it contains this project code
- output: it stores all results produced by the algorithms for each round/run, e.g. "avg_expected_lossses.txt" contains a row for each indipendent cmab run and each row contains, for each t=0,...,T-1 (separated with ";"), the expected cumulative (up to round t) average disagreement of the yielded clusterings.
  - 

## Usage

From the folder 'CMAB-CC/code', run the following command:
```bash      

run_CMAB.py [-h] -d DATASET_NAME [-b {cc-clcb,cc-clcb-m,global-clcb,global-clcb-m,eg,eg-fixed,pe,cts,pcexp-clcb}] [-eps EXPLORATION_PROBABILITY] [-o {pivot,charikar}] [-T TIMESTEPS]
                   [-r RUNS] [-s SEED]                            
```
### Dependencies
- networkx==2.6.3
- numpy==1.22.1
- PuLP==2.6.0
- python_igraph==0.9.9
- scipy==1.7.3


### Positional arguments
```
  -d DATASET, --dataset DATASET
                        Input dataset, whose name identifies a particular subfolder in 'datasets/'
```
### Optional arguments
```                                          
  -h, --help            show this help message and exit
  -b {cc-clcb,cc-clcb-m,global-clcb,global-clcb-m,eg,eg-fixed,pe,cts,pcexp-clcb}, --bandit {cc-clcb,cc-clcb-m,global-clcb,global-clcb-m,eg,eg-fixed,pe,cts,pcexp-clcb}
                        Bandit algorithm (default value cc-clcb)
  -eps EXPLORATION_PROBABILITY, --exploration_probability EXPLORATION_PROBABILITY
                        exploration probability for epsilon-greedy CMAB algorithm (default value 0.1)
  -o {pivot,charikar}, --oracle {pivot,charikar}
                        Oracle to use in each CMAB step (default value pivot)
  -T TIMESTEPS, --timesteps TIMESTEPS
                        Number of timesteps/rounds to run the selected bandit algorithm (default value 400)
  -r RUNS, --runs RUNS  Number of (independent) bandit runs (default value 5)
  -s SEED, --seed SEED  Random generation seed -- for reproducibility (default value 100)  

  Modify the file "constants.py" to change the default values for the parameters.          
                    
```

