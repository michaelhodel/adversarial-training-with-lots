# import packages / libraries
import torch
import skimage.metrics


def SSIM(image_one, image_two):
    """ returns the Structural Similarity Index Measure of two images""" 
    multichannel = image_one.shape[0] != 0
    img_one, img_two = image_one.cpu().clone(), image_two.cpu().clone()
    if multichannel:
        img_one, img_two = img_one.permute(1, 2, 0), img_two.permute(1, 2, 0)
    else:
        img_one, img_two = img_one.squeeze(), img_two.squeeze()
    
    return skimage.metrics.structural_similarity(
        img_one.numpy(), img_two.numpy(),
        data_range=1, multichannel=multichannel
    )


def FOSC(model, originals, adversarials, targets, eps):
    """ returns the First-Order Stationary Criterion of a batch"""
    device = originals.device
    perturbations = adversarials - originals
    adversarials.requires_grad = True
    loss = -torch.nn.CrossEntropyLoss()(
        model(adversarials), targets.to(torch.long)
    )
    loss.backward()
    dot_products = torch.Tensor(adversarials.shape[0]).to(device)
    for i in range(adversarials.shape[0]):
        dot_products[i] = torch.tensordot(
            perturbations[i], adversarials.grad[i], dims=([0, 1, 2], [0, 1, 2])
        )
    norms = torch.norm(adversarials.grad, p=1, dim=(1, 2, 3))
    adversarials.detach_()
    
    return eps * norms - dot_products

