# Training 
Each of the experiment configurations follows the same structure, i.e., it is contained in a folder with the name of 
the dataset (e.g., CIFAR10 / Imagenet) and then in a subfolder to cluster types of experiments (with different architectures for example).

In order to run an experiment, you can use the following command: 
```
python experiments/CIFAR10/final/model.py --single_epoch=true --experiment_name=9L-S-CoDA-SQ-1000
``` 
This would run the experiment `9L-S-CoDA-SQ-1000`specified in `experiments/CIFAR10/final/experiment_parameters.py` for a single epoch.
In order to pick up from a previous checkpoint, use the following:  
```
python experiments/CIFAR10/final/model.py --continue_exp=true --experiment_name=9L-S-CoDA-SQ-1000
``` 
Per default, the results are saved in the current directory. In order to specify an output directory, use the `--base_path` option.

For more details, check the argument parser that is used in `experiments/CIFAR10/final/model.py`

In order to train the models according to the same setup used in the publication, run
```
python experiments/CIFAR10/final/model.py --experiment_name=9L-S-CoDA-SQ-1000
``` 
This should reproduce (up to randomness due to initialisation and data sampling) the results presented in our paper.
