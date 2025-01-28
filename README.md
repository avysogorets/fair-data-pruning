# DroP: Distributionally Robust Data Pruning

Code to reproduce experimental results of the paper.

### Quick Setup
Requires Python 3+.
1. Create a conda environment: ```conda env create -f environment.yml```,
2. Activate the environment: ```conda activate environment```.

### Usage
The project implements both active learning (AL, ```--strategy 0```) and data pruning (DP, ```--strategy 1```).
The command line flag ```--auto_config``` fills in the appropriate hyperparameters based on the model (recommended). The workflow of the main script is as follows:
1. Train a query model (possibly across multiple initializations) and retrieves sample scores;
2. Acquire (for AL) or remove (for DP) samples based on scores and other factors (e.g., class-wise quotas); 
3. Potentially repeat steps 1-2 across multiple iterations (```--iterations```, common for AL);
4. Once the ultimate dataset is determined, train the final model and save its metrics in a json format.  

### Examples
Here are a few simple usage examples. The commands should be executed from a parent directory of the project folder.<br/>
- Prune 30% of CIFAR-10 using VGG-16 and EL2N scorer:<br/>
```python -m drop-data-pruning.main --auto_config --use_gpu --strategy 1 --final_frac 0.7 --model_name VGG16 --scorer_name EL2N```<br/>
- Randomly prune 30% of CIFAR-10 using VGG-16 and DRoP class-wise ratios with query retrained 5 times:<br/>
```python -m drop-data-pruning.main --auto_config --use_gpu --strategy 1 --final_frac 0.7 --model_name VGG16 --scorer_name Random --quoter_name DRoP --num_inits 5```<br/>
- Prune 30% of CIFAR-10 using VGG-16 and Forgetting, and train the final model with a cost-sensitive optimization algorithm CDB-W :<br/>
```python -m drop-data-pruning.main --auto_config --use_gpu --cdbw_final --strategy 1 --final_frac 0.7 --model_name VGG16 --scorer_name Random```<br/>

### Cite us
```
@InProceedings{vysogorets2025drop,
title = {DRoP: Distributionally Robust Data Pruning},
author = {Vysogorets, Artem and Ahuja, Kartik and Kempe, Julia},
booktitle = {Proceedings of the 13th International Conference on Learning Representations},
pages = {1--25},
year = {2025},
series = {Proceedings of Machine Learning Research},
month = {24--28 Apr},
publisher = {PMLR}}
```
