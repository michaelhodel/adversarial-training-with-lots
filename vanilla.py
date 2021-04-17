# import packages / libraries
import torch
import tqdm
import pandas
import numpy

# import functions
from attacking import create_adversarials
from attacking import get_random_target_classes
from attacking import get_lots_targets
from evaluation import evaluate


def get_vanilla_data(model, data, num_classes):
    """ returns only correctly predicted pairs and random targets """
    vanilla_predictions = torch.max(model(data.tensors[0]), dim=1)[1]
    correct_predictions_mask = vanilla_predictions == data.tensors[1]
    vanilla_data = torch.utils.data.TensorDataset(
        data.tensors[0][correct_predictions_mask],
        data.tensors[1][correct_predictions_mask],
    )
    numpy.random.seed(42)
    target_classes = get_random_target_classes(
        num_classes, vanilla_data.tensors[1]
    )
    
    return vanilla_data, target_classes


def get_vanilla_adversarials(model, settings, data, targets, data_name,
                             compute, device, lots_targets):
    """ creates adversarials for each setting on the same (base) model """
    vanilla_adversarials = dict()
    iterator = tqdm.tqdm(settings.items(), position=0, leave=True)
    iterator_description = '{}: loading / computing adversarials for {}'
    for name, setting in iterator:
        path = 'adversarials/{}/'.format(data_name)
        iterator.set_description(iterator_description.format(data_name, name))
        if compute:
            vanilla_adversarials[name] = create_adversarials(
                model=model, data=data, config=dict(setting), name=name,
                save=True, path=path, target_classes=targets,
                lots_targets=lots_targets
            )
        else:
            vanilla_adversarials[name] = {
                key: torch.load('{}{}_adversarial_{}.pt'.format(
                    path, name, key
                )).to(device) for key in ['images', 'labels', 'targets']
            }   
    
    return vanilla_adversarials


def evaluate_vanilla_adversarials(model, adversarials, device):
    """ evaluates adversarials (created on base model) """
    vanilla_accuracies = pandas.DataFrame(
        index=['accuracy'], columns=adversarials.keys()
    )   
    vanilla_success_rates = pandas.DataFrame(
        index=['success-rate'], columns=adversarials.keys()
    )   
    for name, adversarials in adversarials.items():
        vanilla_accuracies[name]['accuracy'] = evaluate(
            data=list(zip(adversarials['images'], adversarials['labels'])),
            model=model, device=device
        )   
        vanilla_success_rates[name]['success-rate'] = evaluate(
            data=list(zip(adversarials['images'], adversarials['targets'])),
            model=model, device=device
        )   
    
    vanilla_accuracies = vanilla_accuracies.astype(float).round(4)
    vanilla_success_rates = vanilla_success_rates.astype(float).round(4)
    
    return vanilla_accuracies, vanilla_success_rates


def run_vanilla_adversarials(settings, models, data_name, num_classes,
                             data, load, lots_templates):
    """ computes, evaluates and returns adversarials created on base model """
    device=data.tensors[0].device

    # use only non-training specific attack settings, include until success
    obsolete_settings = [ 
        'base', 'lots-curriculum', 'lots-dynamic',
        'pgd-curriculum', 'pgd-dynamic'
    ]   
    for setting in obsolete_settings:
        _ = settings.pop(setting)

    # get correctly predicted images and targets
    vanilla_data, vanilla_target_classes = get_vanilla_data(
        models['base'], data, num_classes
    )
    
    # get lots_targets
    lots_targets = get_lots_targets(lots_templates, vanilla_target_classes)
    
    vanilla_adversarials = get_vanilla_adversarials(
        model=models['base'], settings=settings,
        data=vanilla_data, targets=vanilla_target_classes,
        lots_targets=lots_targets, data_name=data_name,
        compute=not load, device=device
    )   

    # compute vanilla accuracies and success rates
    vanilla_accuracies, vanilla_success_rates = evaluate_vanilla_adversarials(
        model=models['base'], device=device, adversarials=vanilla_adversarials
    )   

    # save results
    accuracies_path = 'results/{}/vanilla_accuracies.csv'
    success_rates_path = 'results/{}/vanilla_success_rates.csv'
    vanilla_accuracies.to_csv(accuracies_path.format(data_name))
    vanilla_success_rates.to_csv(success_rates_path.format(data_name))

    # display results
    print('{}: accuracies on vanilla adversarials:'.format(data_name))
    print(vanilla_accuracies.to_markdown())
    print('{}: success rates of vanilla adversarials:'.format(data_name))
    print(vanilla_success_rates.to_markdown())

    return vanilla_data, vanilla_adversarials

