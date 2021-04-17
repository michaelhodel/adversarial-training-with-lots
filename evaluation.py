# import packages / libraries
import torch
import pandas
import tqdm

# import modules
import attacking


def evaluate(model, data, device):
    """ returns the accuracy of the provided model on the provided data """
    _ = model.eval()
    with torch.no_grad():
        correct, total = 0, len(data)
        dataloader = torch.utils.data.DataLoader(data, batch_size=50)
        for images, labels in dataloader:
            if len(images.shape) == 3:
                images = images.unsqueeze(1)
            predictions = torch.max(model(images), dim=1)[1]
            correct += (predictions == labels).sum().item()
    return correct / total


def get_evaluation_data(models, data, num_classes=10):
    """ returns only the examples for which all models predict corectly """
    correct_predictions_masks = []
    base_accuracies = pandas.DataFrame(
        columns=list(models.keys()), index=['accuracy']
    )
    for name, model in models.items():
        mask = torch.max(model(data.tensors[0]), dim=1)[1] == data.tensors[1]
        correct_predictions_masks.append(mask)
        base_accuracies[name]['accuracy'] = (mask.sum() / len(mask)).item()
    data_mask = torch.stack(correct_predictions_masks).T.all(dim=1)
    device = data.tensors[0].device
    target_classes = attacking.get_random_target_classes(
        num_classes, data.tensors[1][data_mask]
    )
    evaluation_data = torch.utils.data.TensorDataset(
        data.tensors[0].clone()[data_mask],
        data.tensors[1].clone()[data_mask]
    )
    return evaluation_data, target_classes, base_accuracies


def cross_evaluate(models, attacks, data, target_classes, lots_targets):
    """ evaluates accuracy and success rate for each attack and model """
    device = data.tensors[0].device
    accuracies = pandas.DataFrame(
        columns=list(attacks.keys()), index=list(models.keys())
    )
    success_rates = pandas.DataFrame(
        columns=list(attacks.keys()), index=list(models.keys())
    )
    
    iterator = tqdm.tqdm(attacks.items(), position=0, leave=True)
    for setting, config in iterator:
        iterator.set_description('evaluating {} adversarials'.format(setting))
        for model_name, model in models.items():
            adversarials = attacking.create_adversarials(
                model=model, data=data, name=setting,
                config=dict(config), target_classes=target_classes,
                lots_targets=lots_targets, save=False, path=None
            )
            accuracies[setting][model_name] = evaluate(
                model=model, device=device,
                data=list(zip(adversarials['images'], adversarials['labels']))
            )
            success_rates[setting][model_name] = evaluate(
                model=model, device=device,
                data=list(zip(adversarials['images'], adversarials['targets']))
            )
    return accuracies, success_rates


def run_cross_evaluation(settings, models, data_name, data,
                         num_classes, load, lots_templates):
    """ computes accuracies and success rates for model-attack combinations """
    
    # remove vanilla, curriculum, dynamic and until success settings
    obsolete_settings = [ 
        'base', 'lots-curriculum', 'lots-dynamic', 'lots-until-success',
        'pgd-curriculum', 'pgd-dynamic', 'pgd-until-success'
    ]   
    for setting in obsolete_settings:
        _ = settings.pop(setting)
        
    # get the evaluation data and fixed targets for
    evaluation_data, targets, base_accuracies = get_evaluation_data(
        models, data, num_classes
    )

    # get the targets for lots attacks
    lots_targets = attacking.get_lots_targets(lots_templates, targets)
        
    base_accuracies = base_accuracies.astype(float).round(4)
        
    # save the results
    base_accuracies_path = 'results/{}/base_accuracies.csv'.format(data_name)
    base_accuracies.to_csv(base_accuracies_path)
    
    # print the accuracies on the original test sets
    print('{}: accuracies on original test set:'.format(data_name))
    print(base_accuracies.to_markdown())
        
    # print the share of images correctly predicted by all models
    base_data_fraction = len(evaluation_data) / len(data)
    base_data_percentage = round(base_data_fraction * 100, 2)
    
    print('{}: {} % of images correctly predicted by all models'.format(
        data_name, base_data_percentage
    ))  
       
    accuracies_path = 'results/{}/accuracies.csv'.format(data_name)
    success_rates_path = 'results/{}/success_rates.csv'.format(data_name)
    
    if not load:
        # perform cross-evaluation
        accuracies, success_rates = cross_evaluate(
            models=models, attacks=dict(settings),
            data=evaluation_data, target_classes=targets,
            lots_targets=lots_targets
        )
    
        # save cross-evaluation results to .csv files
        accuracies = accuracies.astype(float).round(4)
        success_rates = success_rates.astype(float).round(4)
        accuracies.to_csv(accuracies_path)
        success_rates.to_csv(success_rates_path)
    
    else:
        # load cross-evaluation results to pandas dataframes
        accuracies = pandas.read_csv(accuracies_path, index_col=0)
        success_rates = pandas.read_csv(success_rates_path, index_col=0)
    
    # display results tables
    print('{}: cross-evaluation accuracies'.format(data_name))
    print(accuracies.to_markdown())
    print('{}: cross-evaluation success rates'.format(data_name))
    print(success_rates.to_markdown())

