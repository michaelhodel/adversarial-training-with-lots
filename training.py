# import packages / libraries
import numpy
import torch
import tqdm
from os.path import exists
from os import mkdir

# import functions / classes
from attacking import attack, TemplatesHandler
from models import MNIST_classifier, CIFAR10_classifier, IMAGENET_classifier
from helpers import load_yaml


model_class_mapper = { 
    'mnist': MNIST_classifier,
    'cifar10': CIFAR10_classifier,
    'imagenet': IMAGENET_classifier
}


def fit(model, optimizer, training_data, n_epochs, batch_size,
        seed, adversarial_config, data_name):
    """ fits the model, including adversarial training if specified """
    device = training_data.tensors[0].device
    numpy.random.seed(seed)
    num_classes = model.num_classes
    dataloader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )   
    epochs = tqdm.tqdm(range(n_epochs), position=0, leave=True)
    mean_train_losses = []
    
    # initialize templates handler if lots adversarial training
    if adversarial_config['attack_type'] == 'lots':
        templates_handler = TemplatesHandler(
            template_size=adversarial_config.pop('template_size'),
            num_classes=num_classes, device=device
        )   
    else:
        templates_handler = None
    
    # iterate over the epochs
    for epoch in epochs:
        training_loss, batch_count = 0, 0
        _ = torch.manual_seed(seed)
        
        # set epoch specific hyperparameters for attacks
        if adversarial_config['attack_type'] is not None:
            if adversarial_config['training_type'] == 'dynamic':
                max_fosc = adversarial_config['max_fosc']
                control_epoch = int(0.8 * n_epochs)
                fosc = max(max_fosc - epoch * max_fosc / control_epoch, 0)
                adversarial_config['fosc'] = fosc
                adversarial_config[
                    'additional_stopping_criterion'
                ] = 'fosc_threshold'
            if adversarial_config['training_type'] == 'curriculum':
                eps = adversarial_config['max_eps'] * epoch / (n_epochs - 1)
                nb_iter = epoch
                adversarial_config['eps'] = eps 
                adversarial_config['nb_iter'] = nb_iter
        
        # iterate over the batches
        for images, labels in dataloader:
            images, labels = images.clone().to(device), labels.to(device)
            train_on = []
            attack_type = adversarial_config['attack_type']
            if attack_type is None or adversarial_config['append']:
                train_on.append(images)
            
            # compute adversarial images
            if attack_type is not None:
                adversarials, _ = attack(
                    model=model, images=images, labels=labels,
                    config=dict(adversarial_config), round_values=False,
                    templates_handler=templates_handler,
                    epoch=epoch, data_name=data_name
                )   
                train_on.append(adversarials)
            
            # update the network parameters
            for mini_batch in train_on:
                mini_batch.requires_grad = True
                loss = torch.nn.CrossEntropyLoss()(model(mini_batch), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
                mini_batch.requires_grad = False
            batch_count += 1
        mean_train_loss = training_loss / (batch_count * len(train_on))
        mean_train_losses.append(mean_train_loss)
        epochs.set_description(
            'Epoch: {}, Loss: {}'.format(epoch + 1, mean_train_loss)
        )   
    
    return torch.Tensor(mean_train_losses)


def fit_and_save(model_class, seed, training_config, name,
                 adversarial_config, training_data, data_name):
    """ fits the model, saves the model parameters and returns the model """
    device = training_data.tensors[0].device
    model = model_class(seed)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), **training_config['optimizer']
    )
    losses = fit(
        model=model, optimizer=optimizer, training_data=training_data,
        adversarial_config=adversarial_config,
        **training_config['fit'], seed=seed, data_name=data_name
    )
    path = 'results/{}/models/{}_classifier.pt'.format(data_name,  name)
    torch.save(model.state_dict(), path)
    losses_path = 'results/{}/losses/{}_losses.pt'.format(data_name, name)
    torch.save(losses, losses_path)
    
    return model, losses


def fit_or_load_models(settings, data_name, seed, data, fit):
    """ either fits and saves or loads all models """
    device = data.tensors[0].device
    
    # get model class
    model = model_class_mapper[data_name]
    
    # load training hyperparameters
    training_config = load_yaml('configs/training/{}.yaml'.format(data_name))

    # not considering until success attack settings
    _ = settings.pop('lots-until-success')
    _ = settings.pop('pgd-until-success')

    # fit or load models
    models, losses = dict(), dict()

    for attack_name, attack_config in settings.items():
        if fit:
            print('fitting {} model using {}'.format(data_name, attack_name))
            models[attack_name], losses[attack_name] = fit_and_save(
                name=attack_name, model_class=model, seed=seed,
                training_config=dict(training_config),
                adversarial_config=dict(attack_config), training_data=data,
                data_name=data_name
            )
        else:
            models[attack_name] = model(seed).to(device)
            model_path = 'results/{}/models/{}_classifier.pt'.format(
                data_name, attack_name
            )
            _ = models[attack_name].load_state_dict(torch.load(model_path))
            losses_path = 'results/{}/losses/{}_losses.pt'.format(
                data_name, attack_name
            )
            losses[attack_name] = torch.load(losses_path)
        _ = models[attack_name].eval()

    return models, losses

