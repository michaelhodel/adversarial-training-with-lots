# import packages / libraries
import torch
import pandas
import tqdm

# import functions
from evaluation import evaluate
from attacking import create_adversarials
from attacking import get_random_target_classes
from attacking import get_lots_targets


def check_transferability(models, attacks, data_name, data,
                          target_classes, lots_templates):
    """ evaluates each model in the models on each dataset in the datasets """
    device = data.tensors[0].device
    base_success_rates = pandas.DataFrame(
        columns=list(attacks.keys()), index=['success rate']
    )   
    transferability_rates = pandas.DataFrame(
        columns=list(attacks.keys()), index=list(models.keys())
    )   
    
    iterator = tqdm.tqdm(attacks.items(), position=0, leave=True)
    for setting, config in iterator:
        iterator.set_description(
            '{}: checking transferability for {}'.format(data_name, setting)
        )   
        logits = models[setting](data.tensors[0])
        predicted_labels = torch.max(logits, dim=1)[1]
        successful_prediction_mask = predicted_labels == data.tensors[1]
        data_subset = torch.utils.data.TensorDataset(
            data.tensors[0].clone()[successful_prediction_mask],
            data.tensors[1].clone()[successful_prediction_mask]
        )
        target_classes_subset = target_classes.clone()[
            successful_prediction_mask
        ]
        lots_targets = get_lots_targets(lots_templates, target_classes_subset)
        adversarials = create_adversarials(
            model=models[setting], data=data_subset, name=setting,
            config=dict(config), target_classes=target_classes_subset,
            lots_targets=lots_targets, save=False, path=None
        )
        logits = models[setting](adversarials['images'])	
        predicted_labels = torch.max(logits, dim=1)[1]
        successful_attacks_mask = predicted_labels == adversarials['targets']
        base_success_rates[setting]['success rate'] = (
            successful_attacks_mask.sum() / len(successful_attacks_mask)
        ).item()
        transferability_data = torch.utils.data.TensorDataset(
            adversarials['images'].clone()[successful_attacks_mask],
            adversarials['targets'].clone()[successful_attacks_mask]
        )   
        for model_name, model in models.items():
            transferability_rates[setting][model_name] = evaluate(
                model=model, device=device, data=transferability_data
            )   
    iterator.set_description('{}: checking transferability'.format(data_name))
    
    return transferability_rates, base_success_rates


def run_transferability_experiments(settings, models, data_name, num_classes,
                                    data, lots_templates):
    """ checks transferability, for varied defense settings """
    obsolete_settings = [ 
        'base', 'lots-curriculum', 'lots-dynamic', 'lots-until-success',
        'pgd-curriculum', 'pgd-dynamic', 'pgd-until-success'
    ]   
    for setting in obsolete_settings:
        _ = settings.pop(setting)
    
    # check transferability
    transferability_target_classes = get_random_target_classes(
        num_classes=num_classes, labels=data.tensors[1]
    )   
    
    transferability_rates, base_success_rates = check_transferability(
        models=models, attacks=settings, data_name=data_name,
        data=data, target_classes=transferability_target_classes,
        lots_templates=lots_templates
    )   
    
    transferability_rates = transferability_rates.astype(float).round(4)
    path = 'results/{}/transferability_rates.csv'.format(data_name)
    transferability_rates.to_csv(path)

    # display results
    print('{}: transferability rates:'.format(data_name))
    print(transferability_rates.to_markdown())

