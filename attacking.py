# import packages / libraries
import numpy
import torch

# import functions
from metrics import FOSC

    
class TemplatesHandler():
    """ handles target updating and retrieving for the LOTS attack """
    def __init__(self, template_size, num_classes, device):
        self.templates = {i: [] for i in range(num_classes)}
        self.template_size = template_size
        self.device = device
    
    def add_to_templates(self, labels, representations):
        """ adds a representation as a target """
        for label, representation in zip(labels, representations):
            label = label.item()
            self.templates[label].append(representation)
            if len(self.templates[label]) > self.template_size:
                self.templates[label] = self.templates[label][1:]
    
    def get_targets(self, labels, target_classes, target_shape):
        """ returns a target representation """
        targets = []
        for label, target_class in zip(labels, target_classes):
            label, target_class = label.item(), target_class.item()
            if len(self.templates[target_class]) == 0:
                target = torch.rand(target_shape)
            else:
                images_sum = sum(self.templates[target_class])
                target = images_sum / len(self.templates[target_class])
            targets.append(target.to(self.device))
        return torch.stack(targets).detach().to(self.device)


def get_lots_templates(model, images, labels, num_classes=10):
    """ forms templates for evaluation """
    with torch.no_grad():
        logits = model(images)
    correct_predictions_mask = torch.max(logits, dim=1)[1] == labels
    class_logits = {i: [] for i in range(num_classes)}
    for i in range(len(labels)):
        if correct_predictions_mask[i].item():
            class_logits[labels[i].item()].append(logits[i])
    
    return {i: sum(t) / len(t) for i, t in class_logits.items()}


def get_lots_targets(templates, target_classes):
    """ returns targets from templates and given target classes """
    return torch.stack([templates[i.item()] for i in target_classes])


def get_random_target_classes(num_classes, labels):
    """ returns a tensor of random target classes """
    label_list, target_classes = set(range(num_classes)), []
    for label in labels:
        available_targets = list(label_list - {label.item()})
        target_classes.append(numpy.random.choice(available_targets))
    
    return torch.tensor(target_classes, device=labels.device)


def perturb(images, adversarials, logits, targets, gradient_treatment,
            step_size, p, eps, control_vector, iterations):
    """ perturbs a batch of images to decrease loss w.r.t. targets """
    p = float(p)
    device = images.device
    if gradient_treatment == 'max_scaling':
        adversarial_loss = torch.nn.MSELoss()(logits, targets)
    elif gradient_treatment == 'sign':
        adversarial_loss = torch.nn.CrossEntropyLoss()(logits, targets)
    else:
        raise NotImplementedError('only max scaling and sign.')
    adversarial_loss.backward()
    gradients = adversarials.grad.detach()
    if p == float('inf'):
        perturbations = torch.zeros(adversarials.shape).to(device)
        for i, indicator in enumerate(control_vector):
            if indicator.item():
                if gradient_treatment == 'max_scaling':    
                    perturbations[i] = gradients[i] / gradients[i].abs().max()
                elif gradient_treatment == 'sign':
                    perturbations[i] = gradients[i].sign()
                else:
                    raise NotImplementedError('only max scaling and sign.')
        adversarials = adversarials - step_size * perturbations
        if eps != 'none' and step_size * iterations > eps:
            eps = float(eps)
            adversarials = adversarials.min(images + eps).max(images - eps)
        adversarials = torch.clamp(adversarials, 0, 1)
    else:
        raise NotImplementedError('currently only supporting max norm')
    
    return adversarials.detach()


def attack(model, images, labels, config, round_values, templates_handler=None,
           target_classes=None, epoch=None, data_name=None, lots_targets=None):
    """ creates adversarial examples using the specified attack """
    attack_type = config.pop('attack_type')
    num_classes = model.num_classes
    if target_classes is None:
        target_classes = get_random_target_classes(num_classes, labels)
    adversarials = images.clone().detach()

    # handle attack type
    logits = model(adversarials)
    if attack_type == 'pgd':
        targets = target_classes
        gradient_treatment = 'sign'
    elif attack_type == 'lots':
        if templates_handler is not None:
            targets = templates_handler.get_targets(	
                labels, target_classes, logits[0].shape
            )
            templates_handler.add_to_templates(	
                labels=labels, representations=logits.clone().detach()	
            )
        else:
            targets = lots_targets
        gradient_treatment = 'max_scaling'
    else:
        raise NotImplementedError('Only LOTS and PGD attacks are implemented.')
    
    # handle stopping criterion via control vector
    additional_stopping_criterion = 'none'
    control_vector = torch.ones(images.shape[0]).bool()
    if 'additional_stopping_criterion' in config:
        additional_stopping_criterion = config['additional_stopping_criterion']
        if additional_stopping_criterion == 'success':
            with torch.no_grad():
                predicted_labels = torch.max(model(adversarials), dim=1)[1]
            control_vector = predicted_labels != target_classes
    
    # perturb until stopping criterion is met
    iterations, nb_iter = 0, float(config['nb_iter'])
    while control_vector.any() and iterations < nb_iter:
        adversarials.requires_grad = True
        logits = model(adversarials)
        adversarials = perturb(
            images=images, adversarials=adversarials,
            logits=logits, targets=targets,
            gradient_treatment=gradient_treatment, p=config['p'],
            eps=config['eps'], step_size=config['step_size'],
            control_vector=control_vector, iterations=iterations
        )
        if additional_stopping_criterion == 'success':
            rounded_adversarials = adversarials.mul(255.0).round().div(255.0)
            with torch.no_grad():
                predicted_labels = torch.max(
                    model(rounded_adversarials), dim=1
                )[1]
            control_vector = predicted_labels != target_classes
        elif additional_stopping_criterion == 'fosc_threshold':
            fosc_values = FOSC(
                model=model, originals=images, adversarials=adversarials,
                targets=target_classes, eps=config['eps']
            )
            path = 'results/{}/fosc_values/{}'.format(data_name, attack_type)
            torch.save(fosc_values, '{}/{}.pt'.format(path, epoch))
            control_vector = fosc_values >= config['fosc']
        iterations += 1
    if round_values:
        adversarials = adversarials.mul(255.0).round().div(255.0)
    
    return adversarials.detach(), target_classes


def create_adversarials(model, data, config, name, path, save,
                        target_classes, lots_targets):
    """ creates and saves adversarials given a model, dataset and attack """
    num_classes, device = model.num_classes, data.tensors[0].device
    images = data.tensors[0].to(device)
    labels = data.tensors[1].to(device)
    adversarial_images, _ = attack(
        model=model, images=images, labels=labels, config=dict(config),
        round_values=True, target_classes=target_classes,
        lots_targets=lots_targets
    )
    adversarials = {
        'images': adversarial_images,
        'labels': labels,
        'targets': target_classes
    }
    if save:
        for tensor_name in adversarials.keys():
            filename = '{}_adversarial_{}.pt'.format(name, tensor_name)
            file_path = '{}{}'.format(path, filename)
            torch.save(adversarials[tensor_name], file_path)
    
    return adversarials

