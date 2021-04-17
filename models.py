# import packages / libraries
import torch
from torchvision.models import resnet


class MNIST_classifier(torch.nn.Module):
    """ implements a simple ConvNet for classifying MNIST images """
    
    def __init__(self, seed):
        """ initializes two Conv-Layers followed by two linear layers """
        super().__init__()
        _ = torch.manual_seed(seed)

        self.layers = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 3, kernel_size=3, padding=1),
            torch.nn.Conv2d(3, 9, kernel_size=3, padding=1),
            torch.nn.Linear(in_features=441, out_features=66),
            torch.nn.Linear(in_features=66, out_features=10)
        ])

        self.batch_norm_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm2d(num_features=3),
            torch.nn.BatchNorm2d(num_features=9)
        ])

        self.num_conv_layers = 2
        self.num_linear_layers = 2
        self.num_classes = 10

    def forward(self, x):
        """ passes the input through the network and returns the output """
        for i, layer in enumerate(self.layers):
            if i == self.num_conv_layers:
                x = x.flatten(start_dim=1)
            x = layer(x)
            if i < self.num_conv_layers + self.num_linear_layers:
                x = torch.nn.ReLU()(x)
            if i < self.num_conv_layers:
                x = self.batch_norm_layers[i](x)
                x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        return x


class CIFAR10_classifier(torch.nn.Module):
    """ implements a simple ConvNet for classifying CIFAR-10 images """
    
    def __init__(self, seed):
        """ initializes six Conv-Layers followed by two linear layers """
        super().__init__()
        _ = torch.manual_seed(seed)

        self.layers = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 6, kernel_size=3, padding=1),
            torch.nn.Conv2d(6, 9, kernel_size=3, padding=1),
            torch.nn.Conv2d(9, 12, kernel_size=3, padding=1),
            torch.nn.Conv2d(12, 15, kernel_size=3, padding=1),
            torch.nn.Conv2d(15, 18, kernel_size=3, padding=1),
            torch.nn.Conv2d(18, 21, kernel_size=3, padding=1),
            torch.nn.Linear(in_features=336, out_features=58),
            torch.nn.Linear(in_features=58, out_features=10)
        ])

        self.batch_norm_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm2d(num_features=6),
            torch.nn.BatchNorm2d(num_features=9),
            torch.nn.BatchNorm2d(num_features=12),
            torch.nn.BatchNorm2d(num_features=15),
            torch.nn.BatchNorm2d(num_features=18),
            torch.nn.BatchNorm2d(num_features=21)
        ])

        self.num_classes = 10
        self.num_conv_layers = 6
        self.num_linear_layers = 2

    def forward(self, x):
        """ passes the input through the network and returns the output """
        for i, layer in enumerate(self.layers):
            if i == self.num_conv_layers:
                x = x.flatten(start_dim=1)
            x = layer(x)
            if i < self.num_conv_layers + self.num_linear_layers:
                x = torch.nn.ReLU()(x)
            if i < self.num_conv_layers:
                x = self.batch_norm_layers[i](x)
                if i % 2 == 1:
                    x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        return x


class IMAGENET_classifier(resnet.ResNet):
    """ adjusted version of the ResNet18 for fewer classes """
    def __init__(self, seed, num_classes=10):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2])
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(
            in_features=self.fc.in_features,
            out_features=self.num_classes
        )

