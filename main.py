# import packages / libraries
import torch
import numpy
from copy import deepcopy

# import functions
from helpers import load_attack_settings
from helpers import load_and_preprocess_data
from helpers import create_directories_if_nonexistent
from training import fit_or_load_models
from evaluation import run_cross_evaluation
from transferability import run_transferability_experiments
from vanilla import run_vanilla_adversarials
from visualizing import run_visualizations
from attacking import get_lots_templates

# use cuda if avaiable
GPU_index = 4
GPU = 'cuda:{}'.format(GPU_index)
device = torch.device(GPU if torch.cuda.is_available() else 'cpu')
print('using device', device)

# names of datasets for which to run the experiments
data_names = ['mnist', 'cifar10']

# indicators whether to compute or load results
fit_models = False
compute_vanilla_adversarials = False
perform_cross_evaluation = False

# specify and set a seed
seed = 42
_ = torch.manual_seed(seed)
numpy.random.seed(seed)

# load attack settings
all_attack_settings = load_attack_settings()

# make directories if nonexistent
create_directories_if_nonexistent(data_names)


def run_experiments(data_name):
    """" runs all experiments on a given dataset """
    print('Running experiments on {} data'.format(data_name))

    # load data
    train_data, test_data, classes = load_and_preprocess_data(
        data_name=data_name,
        device=device
    )
    
    # fit or load models
    models_dict, losses = fit_or_load_models(
        settings=deepcopy(all_attack_settings),
        data_name=data_name,
        seed=seed,
        data=train_data,
        fit=fit_models
    )
    
    # get fixed templates for evaluating lots attacks
    lots_templates = get_lots_templates(
        model=models_dict['base'],
        images=train_data.tensors[0],
        labels=train_data.tensors[1]
    )
    
    # vanilla adversarials
    vanilla_data, vanilla_adversarials = run_vanilla_adversarials(
        settings=deepcopy(all_attack_settings),
        models=models_dict,
        data_name=data_name,
        num_classes=len(classes),
        data=test_data,
        load=not compute_vanilla_adversarials,
        lots_templates=lots_templates
    )   
    
    # cross-evaluation
    run_cross_evaluation(
        settings=deepcopy(all_attack_settings),
        models=models_dict,
        data_name=data_name,
        data=test_data,
        num_classes=len(classes),
        load=not perform_cross_evaluation,
        lots_templates=lots_templates
    )
    
    # transferability
    run_transferability_experiments(
        settings=deepcopy(all_attack_settings),
        models=models_dict,
        data_name=data_name,
        num_classes=len(classes),
        data=test_data,
        lots_templates=lots_templates
    )
    
    # visualizations
    run_visualizations(
        vanilla_adversarials=vanilla_adversarials,
        vanilla_data=vanilla_data,
        training_data=train_data,
        data_name=data_name,
        losses=losses,
        model=models_dict['base'],
        classes=classes
    )
    
    print('Finished experiments on {} data'.format(data_name))


# run experiments
for data_name in data_names:
    run_experiments(data_name)

