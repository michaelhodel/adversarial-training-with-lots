# import packages / libraries
import yaml
import tqdm
import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from PIL import Image
from os import listdir, mkdir
from os.path import exists


def load_yaml(path):
    """ loads a .yaml file and returns a dictionary """
    with open(path, 'r') as yaml_file:
        return yaml.load(yaml_file, yaml.FullLoader)


def load_attack_settings():
    """ returns a dictionary of all attack settings """
    all_attack_settings = {'base': {'attack_type' : None}}
    attack_settings_names = [ 
        'lots-single', 'lots-10', 'lots-curriculum', 'lots-dynamic',
        'lots-until-success', 'pgd-single', 'pgd-10', 'pgd-curriculum',
        'pgd-dynamic', 'pgd-until-success',
    ]   
    
    for attack_setting_name in attack_settings_names:
        all_attack_settings[attack_setting_name] = load_yaml(
            'configs/attacking/{}.yaml'.format(attack_setting_name)
        )   
    
    return all_attack_settings


def create_directories_if_nonexistent(data_names):
    """ creates all directories needed to save results """
    for d in ['data', 'results']:
        if not exists(d):
            mkdir(d)
    directories = ['models', 'fosc_values', 'losses', 'plots']
    for data_name in data_names:
        data_path = 'results/{}'.format(data_name)
        if not exists(data_path):
            mkdir(data_path)
        for directory in directories:
            path = '{}/{}'.format(data_path, directory)
            if not exists(path):
                mkdir(path)
        for attack_type in ['lots', 'pgd']:
            path = 'results/{}/fosc_values/{}'.format(
                data_name, attack_type
            )
            if not exists(path):
                mkdir(path)


def load_and_preprocess_mnist_data(device):
    """ loads and preprocesses the MNIST data """
    # load the MNIST datasets
    MNIST_train = MNIST(root='data/mnist/', train=True, download=True)
    MNIST_test = MNIST(root='data/mnist/', train=False, download=True)

    # rescale to range [0, 1] and transform to TensorDataset, extra dim
    MNIST_train_data = torch.utils.data.TensorDataset(
        MNIST_train.data.float().div(255.0).unsqueeze(1).to(device),
        MNIST_train.targets.to(device)
    )   

    MNIST_test_data = torch.utils.data.TensorDataset(
        MNIST_test.data.float().div(255.0).unsqueeze(1).to(device),
        MNIST_test.targets.to(device)
    )
    
    classes = [c.split(' ')[-1] for c in MNIST_train.classes]

    return MNIST_train_data, MNIST_test_data, classes


def load_and_preprocess_cifar10_data(device):
    """ loads and preprocesses the CIFAR-10 data """
    # load the CIFAR-10 datasets
    CIFAR10_train = CIFAR10(root='data/cifar10/', train=True, download=True)
    CIFAR10_test = CIFAR10(root='data/cifar10/', train=False, download=True)

    # rescale to range [0, 1] transform to TensorDataset, nhwc to nchw
    CIFAR10_train_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(
            CIFAR10_train.data
        ).div(255.0).permute(0, 3, 1, 2).to(device),
        torch.Tensor(CIFAR10_train.targets).to(torch.int64).to(device)
    )

    CIFAR10_test_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(
            CIFAR10_test.data
        ).div(255.0).permute(0, 3, 1, 2).to(device),
        torch.Tensor(CIFAR10_test.targets).to(torch.int64).to(device)
    )

    return CIFAR10_train_data, CIFAR10_test_data, CIFAR10_train.classes


def load_imagenet_data(class_selection, imagenet_path,
                       device, load_from_jpeg, train):
    """ loads the images of a selection of imagenet classes """
    msg = 'Loading ImageNet {} data for selection of {} classes'
    n_classes = len(class_selection.keys())
    print(msg.format('train' if train else 'test', n_classes))
    
    # preprocessing to get uniformly sized tensors
    preprocessor = Compose([Resize(288), CenterCrop(256), ToTensor()])
    dataset_type = 'train' if train else 'val'

    # load from jpeg files or tensors
    data_dict = dict()
    for wordnet_id in class_selection.keys():
        tensor_filename = 'data/imagenet/{}/{}.pt'.format(
            dataset_type, wordnet_id
        )
        if load_from_jpeg:
            class_path = '{}/{}/{}'.format(
                imagenet_path, dataset_type, wordnet_id
            )
            class_image_filenames = listdir(class_path)
            class_images = []
            file_iterator = tqdm.tqdm(
                class_image_filenames, position=0, leave=True
            )
            for filename in file_iterator:
                image_path = '{}/{}'.format(class_path, filename)
                image = preprocessor(Image.open(image_path))
                if image.shape[0] == 3:
                    class_images.append(image)
            data_dict[wordnet_id] = torch.stack(class_images)
            torch.save(data_dict[wordnet_id], tensor_filename)
        else:
            data_dict[wordnet_id] = torch.load(tensor_filename)

    images_list, classes_list = [], []
    
    for i, (wordnet_id, images) in enumerate(data_dict.items()):
        images_list.append(images)
        classes_list.append(torch.full((images.shape[0],), i)) 

    images_tensor = torch.cat(images_list)
    classes_tensor = torch.cat(classes_list)
    torch.manual_seed(42)
    shuffled_order = torch.randperm(len(classes_tensor))
    dataset = torch.utils.data.TensorDataset(
        images_tensor[shuffled_order], classes_tensor[shuffled_order]
    )

    return dataset


def load_and_preprocess_imagenet_data(device, load_from_jpeg=True):
    """ loads and preprocesses the imagenet training and testing data """
    imagenet_path = '/local/scratch/datasets/ImageNet/ILSVRC2012/'

    class_selection = {
        'n02088466': 'bloodhound',
        'n02488702': 'colobus',
        'n02391049': 'zebra',
        'n02105505': 'komondor',
        'n02492035': 'capuchin',
        'n01518878': 'ostrich',
        'n01531178': 'goldfinch',
        'n01818515': 'macaw',
        'n01944390': 'snail',
        'n01910747': 'jellyfish',
    }

    imagenet_train_data = load_imagenet_data(
        class_selection, imagenet_path, device, load_from_jpeg, train=True
    )
    imagenet_test_data = load_imagenet_data(
        class_selection, imagenet_path, device, load_from_jpeg, train=False
    )
    classes = list(class_selection.values())
    
    return imagenet_train_data, imagenet_test_data, classes


def load_and_preprocess_data(data_name, device):
    """ load training and testing data """
    data_mapper = { 
        'mnist': load_and_preprocess_mnist_data,
        'cifar10': load_and_preprocess_cifar10_data,
        'imagenet': load_and_preprocess_imagenet_data
    }

    return data_mapper[data_name](device)

