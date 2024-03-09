# Fair Data Pruning

Code for the paper on robust data pruning.

### Quick Setup
Install the required packages with ```pip install -r requirements.txt```

### Usage
The project implements both active learning (```--strategy 0```) and data pruning (```--strategy 1```).
The command line flag ```--auto_config``` fills in the appropriate hyperparameters based on the model specification and is recommended.

### Example
Here are a few simple usage examples. The commands should be executed from a parent directory of the project's folder.<br/>
- Prune 30% of CIFAR-10 using VGG-16 and EL2N scorer:<br/>
```python -m fair-data-pruning.main --auto_config --use_gpu --strategy 1 --final_frac 0.7 --model_name VGG16 --scorer_name EL2N```<br/>
- Randomly prune 30% of CIFAR-10 using VGG-16 and MetriQ class-wise ratios with query retrained 5 times:<br/>
```python -m fair-data-pruning.main --auto_config --use_gpu --strategy 1 --final_frac 0.7 --model_name VGG16 --scorer_name Random --quoter_name MetriQ --num_inits 5```<br/>
- Prune 30% of CIFAR-10 using VGG-16 using Forgetting and train the final model with CDB-W :<br/>
```python -m fair-data-pruning.main --auto_config --use_gpu --cdbw_final --strategy 1 --final_frac 0.7 --model_name VGG16 --scorer_name Random```<br/>
