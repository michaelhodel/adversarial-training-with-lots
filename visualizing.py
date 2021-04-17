# import packages / libraries
import numpy
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import seaborn
from os import listdir
from copy import deepcopy

# import functions
import metrics


def plot_class_images(data, path, classes):
    """ plots an image for each class """
    sample_images = []
    label_list = data.tensors[1].tolist()
    
    for i in range(len(classes)):
        idx = label_list.index(i)
        sample_images.append(data.tensors[0][idx])
    
    fig, ax = plt.subplots(1, len(classes))
    
    for i, label in enumerate(classes):
        ax[i].set_title(label, fontsize='medium')
        image = sample_images[i].cpu()
        if image.shape[0] == 3:
            ax[i].imshow(image.permute(1, 2, 0))
        else:
            ax[i].imshow(
                image.squeeze(), cmap=plt.cm.get_cmap('gray').reversed()
            )
        ax[i].set_xticks([]), ax[i].set_xlabel('')
        ax[i].set_yticks([]), ax[i].set_ylabel('')
    
    plt.subplots_adjust(wspace=0.25, hspace=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0, 0)
    save_to = '{}class_images.pdf'.format(path)
    plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_image_grid(datasets, path, model, classes):
    """ plots a set of images in a grid """
    # order for and number of sample images
    order = [
        'original', 'lots-single', 'lots-10', 'lots-until-success',
        'pgd-single', 'pgd-10', 'pgd-until-success'
    ]
    
    n_samples = 4
    sample_start_index = numpy.random.randint(
        len(datasets['original']) // 2,
        len(datasets['original']) - n_samples - 1
    )
    sample_images = {
        'original': datasets['original'].tensors[0][
            sample_start_index:sample_start_index + n_samples
        ],
    }
    for name in set(datasets.keys()) - { 'original' }:
        images = datasets[name]['images']
        sample_images[name] = images[
            sample_start_index:sample_start_index + n_samples
        ]

    n_datasets = len(sample_images.keys())
    image_grid, subplots = plt.subplots(
        n_samples, n_datasets, figsize=(n_datasets * 2, n_samples * 2)
    )
    plt.setp(subplots.flat, xlabel='X-label', ylabel='Y-label')
    for ax, name in zip(subplots[0], order):
        ax.annotate(
            name, xy=(0.5, 1), xytext=(0, 25), xycoords='axes fraction',
            textcoords='offset points', size='large',
            ha='center', va='baseline'
        )
    labels = datasets['original'].tensors[1][
        sample_start_index:sample_start_index + n_samples
    ]
    targets = datasets['lots-single']['targets'][
        sample_start_index:sample_start_index + n_samples
    ]
    for ax, label, target in zip(subplots[:, 0], labels, targets):
        annotation = 'Class: {}\nTarget: {}'.format(
            classes[label.item()], classes[target.item()]
        )
        ax.annotate(
            annotation, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label, textcoords='offset points',
            size='large', ha='right', va='center'
        )
    for j, name in enumerate(order):
        for i, image in enumerate(sample_images[name]):
            probabilities = torch.nn.Softmax(dim=1)(model(image.unsqueeze(0)))
            p_label = round(probabilities[0][labels[i]].item(), 2)
            p_target = round(probabilities[0][targets[i]].item(), 2)
            subplots[i, j].set_title(
                'p(C): {}, p(T): {}'.format(p_label, p_target),
                fontsize='medium'
            )
            image = image.cpu()
            if image.shape[0] == 3:
                subplots[i, j].imshow(image.permute(1, 2, 0))
            else:
                subplots[i, j].imshow(
                    image.squeeze(),
                    cmap=plt.cm.get_cmap('gray').reversed()
                )
            subplots[i, j].set_xticks([]), subplots[i, j].set_xlabel('')
            subplots[i, j].set_yticks([]), subplots[i, j].set_ylabel('')
    plt.subplots_adjust(wspace=0, hspace=0.25)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_to = '{}sample_images.pdf'.format(path)
    plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    plt.close()


def cumulative_ssim_plot(data_dict, path):
    """ cum. ssim scores between adversarials and corresponding originals """
    originals = data_dict.pop('original').tensors[0]
    for name, data in data_dict.items():
        ssim_scores = []
        image_pairs = zip(originals, data['images'])
        for original, adversarial in image_pairs:
            ssim_scores.append(metrics.SSIM(original, adversarial))
        counts, bins = numpy.histogram(ssim_scores, bins=1000, range=(0, 1))
        cdf = numpy.cumsum(counts) / numpy.sum(counts)
        plt.plot(
            numpy.vstack((bins, numpy.roll(bins, -1))).T.flatten()[:-2],
            numpy.vstack((cdf, cdf)).T.flatten() * 100
        )
    plt.title('Cumulative SSIM Scores');
    plt.legend(list(data_dict.keys()))
    plt.xlabel('SSIM', size=16);
    plt.ylabel('% below', size=16);
    filename = '{}cumulative_ssim.png'.format(path)
    plt.savefig(filename, transparent=False)
    plt.close()


def euclidean_norm_densities_plot(data_dict, path):
    """ euclidean norms between adversarials and corresponding originals """
    originals = data_dict.pop('original').tensors[0]
    for name, data in data_dict.items():
        euclidean_norms = torch.norm(
            originals - data['images'], p=2, dim=(1, 2, 3)
        )
        seaborn.kdeplot(euclidean_norms.cpu())
    plt.title('Euclidean Norm Distributions');
    plt.legend(list(data_dict.keys()))
    plt.xlabel('P-2', size=16);
    plt.ylabel('density', size=16);
    filename = '{}euclidean_norms.png'.format(path)
    plt.savefig(filename, transparent=False)
    plt.close()


def plot_losses(losses_dict, path):
    """ plots the losses over the epochs for a dictionary of model losses """
    epochs = list(range(1, len(next(iter(losses_dict.values()))) + 1))
    
    color_mapper = {
        'single': 'C0', '10': 'C1', 'curriculum': 'C2', 'dynamic': 'C3'
    }
    linestyle_mapper = {'lots': 'solid', 'pgd': 'dashed'}
    fig, ax = plt.subplots()
    
    for setting, losses in losses_dict.items():
        if setting == 'base':
            linestyle, color = 'solid', 'black'
        else:
            attack, training = setting.split('-')
            color = color_mapper[training]
            linestyle = linestyle_mapper[attack]
        ax.plot(epochs, losses,linestyle=linestyle, color=color)
    
    color_legend_handles = []
    for v in color_mapper.values():
        color_legend_handles.append(Line2D([0], [0], color=v, lw=1))
    ax.legend(
        color_legend_handles, list(color_mapper.keys()),
        loc='upper right'
    )
    
    shape_legend_handles = []
    for v in linestyle_mapper.values():
        shape_legend_handles.append(
            Line2D([0], [0], color='black', lw=1, linestyle=v)
        )
    ax.add_artist(
        Legend(ax, shape_legend_handles,
        list(linestyle_mapper.keys()), loc='upper center')
    )
    
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood Loss')
    filename = '{}training_losses.png'.format(path)
    plt.savefig(filename, transparent=False)
    plt.close()


def plot_fosc_values(data_name, path):
    """ plots the mean fosc values from dynamic training """
    fosc_values_path = 'results/{}/fosc_values'.format(data_name)
    attack_types = ['lots', 'pgd']
    
    for attack in attack_types:
        attack_fosc_values_path = '{}/{}'.format(fosc_values_path, attack)
        filenames = listdir(attack_fosc_values_path)
        mean_fosc_values = []
        for epoch in range(len(filenames)):
            fosc_values = torch.load(
	        '{}/{}.pt'.format(attack_fosc_values_path, epoch)
            )
            mean_fosc_values.append(torch.mean(fosc_values, dim=0).item())
        plt.plot(list(range(1, len(filenames) + 1)), mean_fosc_values)
    
    plt.legend(labels=attack_types)
    plt.title('FOSC Values')
    plt.xlabel('Epoch')
    plt.ylabel('FOSC')
    plot_filename = '{}fosc_values.png'.format(path)
    plt.savefig(plot_filename, transparent=False)
    plt.close()
    

def run_visualizations(vanilla_adversarials, vanilla_data, training_data,
                       data_name, losses, model, classes):
    """ creates and saves all visualizations """
    plots_path = 'results/{}/plots/'.format(data_name)

    # more concise labels
    for i, label in enumerate(classes):
        if label == 'automobile':
            classes[i] = 'car'
        if label == 'airplane':
            classes[i] = 'plane'
    
    # plot an image for each class
    plot_class_images(training_data, plots_path, classes)

    # plotting the losses
    plot_losses(losses, plots_path)
    
    # plot fosc values
    plot_fosc_values(data_name, plots_path)
    
    # plotting some sample images
    plot_image_grid(
        datasets={**vanilla_adversarials, 'original': vanilla_data},
        path=plots_path, model=model, classes=classes
    )

    # structural similarity index measure plots
    until_success_keys = ['lots-until-success', 'pgd-until-success']

    until_success_adversarials =  {
        key: vanilla_adversarials[key] for key in until_success_keys
    }
    
    data_dict = {**until_success_adversarials, 'original': vanilla_data}

    # plot cumulativs ssim values
    cumulative_ssim_plot(
        data_dict=deepcopy(data_dict), path=plots_path
    )
    
    # plot euclidean norm densities
    euclidean_norm_densities_plot(
        data_dict=deepcopy(data_dict), path=plots_path
    )

