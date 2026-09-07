# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
from typing import Optional
import torchvision
import torchvision.transforms as transforms
# -----------------
from .tools import *
from .methods import *
import copy

def create_data(o):
    batch_size = o.batch_size
    torch.manual_seed(o.data_seed)
    # Global datasets
    fname = '../data/' + o.data + '.tensor.pkl'

    try:
        datas = torch.load(fname, weights_only=False)
        print('Loaded cached dataset tensor')
        train_set = datas.train_set
        test_set = datas.test_set
        o.input_shape = datas.input_shape
        o.num_classes = datas.num_classes

    except FileNotFoundError:
        # datasets
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Pytorch "Default"
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]) # "Standartized"
        # training and validation sets

        train_set = Dataset_to_XY(torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform))
        test_set = Dataset_to_XY(torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform))
        o.input_shape = [3, 32, 32]
        o.num_classes = 10
        
        datas = dotdict(train_set=train_set, test_set=test_set, input_shape=o.input_shape, num_classes=o.num_classes)
        torch.save(datas, fname)
        
    train_set, val_set = train_set.split(valid_size=0.1)
    print(f'input_shape= {o.input_shape}, train data range = {train_set.X.flatten().min().item(), train_set.X.flatten().max().item()}, test data range = {test_set.X.flatten().min().item(), test_set.X.flatten().max().item()}')
    # dataloaders
    train_transform = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip()])
    train_set.set_transform(train_transform)
    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True) #TODO: why doesnt work (drop_last)
    
    data = Datas(train_set, train_loader, val_set, val_loader, test_set, test_loader, train_loader_test=train_loader)
    return data  # train_set, train_loader, val_loader, test_set, test_loader


def VGG_Block(o: dotdict, o1: dotdict, in_channels, out_channels, kernel=(6, 6), stride=1, padding=1, avg_kernel=2):
    layers = ESequential(
        QConv2d(o1, in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        QReLU(o, K=o.A),
        QConv2d(o, out_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.MaxPool2d(kernel_size=avg_kernel),
        nn.BatchNorm2d(out_channels),
        QReLU(o, K=o.A)
    )
    return layers


class VGGRQ(EClassificationNet):
    def __init__(self, o):
        o1 = copy.deepcopy(o)
        o1.W = 16  # weights in first and last layers quantized to 16 states = 4 bits
        l = []
        l += [VGG_Block(o, o1, in_channels=3, out_channels=128, kernel=(3, 3), stride=1, padding=1, avg_kernel=2)]
        l += [VGG_Block(o, o, in_channels=128, out_channels=256, kernel=(3, 3), stride=1, padding=1, avg_kernel=2)]
        l += [VGG_Block(o, o, in_channels=256, out_channels=512, kernel=(3, 3), stride=1, padding=0, avg_kernel=2)]
        l += [nn.Flatten()]
        l += [QLinear(o, 2048, 1024, bias=False)]
        l += [nn.BatchNorm1d(num_features=1024)]
        l += [QReLU(o, K=o.A)]
        l += [QLinear(o1, 1024, 10, bias=False)]
        l += [nn.BatchNorm1d(num_features=10, affine=False)]
        super().__init__(*l)
        # init_net(self, 0.5, 0.5)  # adjusting for clipped activation function so that most of the standartized distributino is in the range [0,1]


class CIFAR_AllCNN(EClassificationNet):
    def __init__(self, o, **kwargs):
        super().__init__()
        o1 = copy.deepcopy(o)
        ll = []
        o1.W = 16
        ll += [QConv2d(o1, 3, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
        ll += [nn.BatchNorm2d(96)]
        ll += [QReLU(o, K=o.A)]
        ll += [QConv2d(o, 96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
        ll += [nn.BatchNorm2d(96)]
        ll += [QReLU(o, K=o.A)]
        ll += [QConv2d(o, 96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)]
        ll += [nn.BatchNorm2d(96)]
        ll += [QReLU(o, K=o.A)]
        ll += [QConv2d(o, 96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
        ll += [nn.BatchNorm2d(192)]
        ll += [QReLU(o, K=o.A)]
        ll += [QConv2d(o, 192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
        ll += [nn.BatchNorm2d(192)]
        ll += [QReLU(o, K=o.A)]
        ll += [QConv2d(o, 192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)]
        ll += [nn.BatchNorm2d(192)]
        ll += [QReLU(o, K=o.A)]
        ll += [QConv2d(o, 192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)]
        ll += [nn.BatchNorm2d(192)]
        ll += [QReLU(o, K=o.A)]
        ll += [QConv2d(o, 192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)]
        ll += [nn.BatchNorm2d(192)]
        ll += [QReLU(o, K=o.A)]
        ll += [QConv2d(o1, 192, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)]
        ll += [nn.AdaptiveAvgPool2d((1,1))]
        ll += [nn.Flatten()]
        ll += [ScaleBias(channels=10, bias=True)]
        super().__init__(*ll)
        # init_net(self, 0.5, 0.5)  # adjusting for clipped activation function so that most of the standartized distributino is in the range [0,1]
        # init_net(self, 0.5, 0.1) # New BN Adaptor Layer in front of QReLU?
        init_net(self, 0.1, 0.5)


def make_ReLU(o):
    return QReLU(o, K=o.A)
    # return nn.ReLU()


def make_conv(o, *args, **kwargs):
    return QConv2d(o, *args, **kwargs)
    # return nn.Conv2d(*args, **kwargs)


def make_linear(o, *args, **kwargs):
    return QLinear(o, *args, **kwargs)
    # return nn.Linear(*args, **kwargs)



class ConvBlock(ESequential):
    def __init__(
        self,
        o,
        inplanes: int,
        planes: int,
        stride: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = make_conv(o, inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=(stride,stride), bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = make_ReLU(o)
        self.conv2 = make_conv(o, planes, planes, kernel_size=(3, 3), padding=(1, 1), stride=(1,1), bias = False)
        # self.bn2 = nn.BatchNorm2d(planes)
        

# class Downsample(ESequential):
#     def __init__(
#         self,
#         o,
#         inplanes: int,
#         planes: int,
#     ) -> None:
#         super().__init__()
#         self.conv = QConv2d(o, inplanes, planes, kernel_size=(2, 2), stride=(2, 2), bias=False)
#         self.bn = nn.BatchNorm2d(planes)
#         # super().__init__(self.conv,self.bn)


class BasicBlock(Parallel):
    def __init__(self,
                 o,
                 inplanes: int,
                 planes: int,
                 stride: int = 1
                 ) -> None:
        skip = ESequential()
        if stride > 1:
            skip.append(QConv2d(o, inplanes, planes, kernel_size=(stride, stride), stride=(stride, stride), bias=False))
        skip.append(nn.BatchNorm2d(planes))
        # skip.append(ConstantScaleBias(0.9))
        conv = ConvBlock(o, inplanes, planes, stride)
        conv.append(nn.BatchNorm2d(planes))
        # conv.append(ConstantScaleBias(0.1))
        super().__init__(conv, skip) # parallel reduction with coefficinets 0.9 for identity / skip branch and 0.1 for conv branch, BN affine factors can learn new relative scales

class ResNetLayer(ESequential):
    def __init__(self,
                 o,
                 inplanes: int,
                 planes: int,
                 stride: int = 1
                 ) -> None:
        ll = []
        ll += [BasicBlock(o, inplanes, planes, stride=stride)]
        ll += [nn.BatchNorm2d(planes)]
        ll += [make_ReLU(o)]
        ll += [BasicBlock(o, planes, planes)]
        ll += [nn.BatchNorm2d(planes)]
        ll += [make_ReLU(o)]
        super().__init__(*ll)


class ResNet18(EClassificationNet):
    def __init__(self, o, **kwargs):
        o1 = copy.deepcopy(o)
        o1.W = 16  # weights in first and last layers quantized to 16 states = 4 bits
        ll = []
        
        # ll += [make_conv(o1, 3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
        # ll += [nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
        ll += [nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)]
        ll += [nn.BatchNorm2d(64)]
        ll += [make_ReLU(o)]
        # ll += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        
        # layer1
        ll += [ResNetLayer(o, 64, 64)]
        # layer2
        ll += [ResNetLayer(o, 64, 128, stride=2)]
        # layer3
        ll += [ResNetLayer(o, 128, 256, stride=2)]
        # layer4
        ll += [ResNetLayer(o, 256, 512, stride=2)]

        ll += [nn.AdaptiveAvgPool2d((1,1))]
        ll += [nn.Flatten()]
        # ll += [make_linear(o1, in_features=512, out_features=10, bias=False)]
        ll += [nn.Linear(in_features=512, out_features=10, bias=True)]
        # ll += [ScaleBias(in_channels=10, bias=True)]
        
        super().__init__(*ll)
        # init_net(self, 0.5, 0.1)  # adjusting for clipped activation function so that most of the standartized distributino is in the range [0,1]
        init_net(self, 1.0, 0.0)  # adjusting for clipped activation function so that most of the standartized distributino is in the range [0,1]
        

def create_net(o, print_info=False):
    torch.manual_seed(o.seed)
    if o.net_name == 'ResNet18' or o.net_name == '':
        net = ResNet18(o).to(dev)
    elif o.net_name == 'AllCNN':
        net = CIFAR_AllCNN(o).to(dev)
    elif o.net_name == 'VGGRQ':
        net = VGGRQ(o).to(dev)
    else:
        raise ValueError(f'Unknown network {o.net_name}')
    return net
