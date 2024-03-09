# Fair Data Pruning

Code for the paper on robust data pruning.

### Quick Setup
Requires Python 3+. Install packages by ```pip install -r requirements.txt```

### Usage
The project implements both active learning (AL, ```--strategy 0```) and data pruning (DP, ```--strategy 1```).
The command line flag ```--auto_config``` fills in the appropriate hyperparameters based on the model specification and is recommended. The general flow of an experiment is as follows:
1. Trains a query model (possibly across multiple initializations) and retrieves sample scores;
2. Acquires (for AL) or deletes (for DP) samples based on scores and other factors (e.g., class-wise quotas); 
3. Potentially repeats steps 1-2 across multiple iterations (```--iterations```, common for AL);
4. Once the ultimate dataset is determined, trains the final model and saves its metrics in a json format.  

### Examples
Here are a few simple usage examples. The commands should be executed from a parent directory of the project's folder.<br/>
- Prune 30% of CIFAR-10 using VGG-16 and EL2N scorer:<br/>
```python -m fair-data-pruning.main --auto_config --use_gpu --strategy 1 --final_frac 0.7 --model_name VGG16 --scorer_name EL2N```<br/>
- Randomly prune 30% of CIFAR-10 using VGG-16 and MetriQ class-wise ratios with query retrained 5 times:<br/>
```python -m fair-data-pruning.main --auto_config --use_gpu --strategy 1 --final_frac 0.7 --model_name VGG16 --scorer_name Random --quoter_name MetriQ --num_inits 5```<br/>
- Prune 30% of CIFAR-10 using VGG-16 and Forgetting, and train the final model with a cost-sensitive optimization algorithm CDB-W :<br/>
```python -m fair-data-pruning.main --auto_config --use_gpu --cdbw_final --strategy 1 --final_frac 0.7 --model_name VGG16 --scorer_name Random```<br/>

### Cite us
Coming soon.
