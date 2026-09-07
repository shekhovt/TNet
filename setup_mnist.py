# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
from .tools import *
import torchvision
import torchvision.transforms as transforms
import copy
import urllib.request
import urllib
from scipy.io import loadmat

from .methods import *
from .tools import force_path

Loader = torch.utils.data.DataLoader

# def binarize_datas(datas: Datas):
#     datas.train_set.binarize()
#     datas.val_set.binarize()
#     datas.test_set.binarize()


class StochastocBinarization(nn.Module):
    def forward(self, probs):
        assert (probs.requires_grad is False)
        return probs.bernoulli()


class WrapIter:
    def __init__(self, iter, transform):
        self.iter = iter
        self.transform = transform

    def __next__(self):
        return self.transform(*next(self.iter))


class DynamicBinLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        def transform(x, y):
            return StochastocBinarization().forward(x), y
        return WrapIter(super().__iter__(), transform)


def create_data(o):
    batch_size = o.batch_size
    # using the seed for randomized data splitting
    torch.manual_seed(o.data_seed)
    # Global datasets
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])

    fname = '../data/' + o.data + '.tensor.pkl'
    try:
        datas = torch.load(fname, weights_only=False)
        print('Loaded cached dataset tensor')
        train_set = datas.train_set
        test_set = datas.test_set
        o.input_shape = datas.input_shape
        o.num_classes = datas.num_classes
    except FileNotFoundError:
        print('Recreating dataset tensor')
        if o.data_base == 'FMNIST':
            train_set = Dataset_to_XY(torchvision.datasets.FashionMNIST('../data', download=True, train=True, transform=transform))
            test_set = Dataset_to_XY(torchvision.datasets.FashionMNIST('../data', download=True, train=False, transform=transform))
            o.input_shape = [1, 28, 28]
            o.num_classes = 10
        elif o.data_base == 'MNIST':
            train_set = Dataset_to_XY(torchvision.datasets.MNIST('../data', download=True, train=True, transform=transform))
            test_set = Dataset_to_XY(torchvision.datasets.MNIST('../data', download=True, train=False, transform=transform))
            o.input_shape = [1, 28, 28]
            o.num_classes = 10
        elif o.data_base == 'Omniglot-28':
            pth = "../data/Omniglot-28/"
            force_path(pth)
            name = pth + "data.mat"
            if not os.path.isfile(name):
                urllib.request.urlretrieve("https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat", name)
            data = loadmat(name)

            def pdata(data):
                return torch.tensor(data).swapdims(0, 1).reshape([-1, 1, 28, 28]).to(torch.float32)

            def ptarget(target):
                return torch.tensor(target).swapdims(0, 1).argmax(dim=1).to(torch.int32)
            train_set = DataXY(pdata(data['data']), ptarget(data['target']))
            test_set = DataXY(pdata(data['testdata']), ptarget(data['testtarget']))
            o.input_shape = [1, 28, 28]
            o.num_classes = 50
        elif o.data_base == 'Omniglot':
            if o.data_rescale_to is not None:
                T = transforms.Compose([transforms.ToTensor(), torchvision.transforms.Resize(size=o.data_rescale_to, antialias=True), transforms.Normalize((0.0,), (1.0,))])
                o.input_shape = [1] + list(o.data_rescale_to)
            else:
                T = transform
                o.input_shape = [1, 105, 105]
            train_set = Dataset_to_XY(torchvision.datasets.Omniglot('../data', download=True, background=True, transform=T))
            test_set = Dataset_to_XY(torchvision.datasets.Omniglot('../data', download=True, background=False, transform=T))
            o.num_classes = 1623
        else:
            raise AttributeError(f'Unknown dataset {o.data}')
        datas = dotdict(train_set=train_set, test_set=test_set, input_shape=o.input_shape, num_classes=o.num_classes)
        torch.save(datas, fname)

    # training and validation sets
    train_set, val_set = train_set.split(valid_size=0.1)
    print(f'input_shape= {o.input_shape}, data range = {train_set.X.flatten().min().item(), train_set.X.flatten().max().item()}')
    # dataloaders
    Loader = torch.utils.data.DataLoader
    if o.static_binarization:
        print('Binarizing whole dataset with fixed threshold')
        train_set.binarize()
        val_set.binarize()
        test_set.binarize()
    if o.dynamic_binarization:
        print('Attaching stochastically binarizing loaders (dynamic binarization)')
        Loader = DynamicBinLoader
    train_loader = Loader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = Loader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = Loader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    data = Datas(train_set, train_loader, val_set, val_loader, test_set, test_loader, train_loader_test=train_loader)
    return data


def Block(o: dotdict, in_channels, out_channels, kernel=(6, 6), stride=1, padding=1, avg_kernel=2):
    layers = ESequential(
        QConv2d(o, in_channels, out_channels, kernel_size=kernel, stride=stride, bias=False),
        nn.MaxPool2d(kernel_size=avg_kernel),
        nn.BatchNorm2d(out_channels),
        # nn.AvgPool2d(kernel_size=avg_kernel),
        QReLU(o.QReLU_A, channels=out_channels)
    )
    return layers


class LeNetRQ(ESequential):
    def __init__(self, o):
        o1 = copy.deepcopy(o)
        o1.QReLU_W = o1.QReLU_W.replace(K=16)  # weights in first and last layers quantized to 16 states = 4 bits
        l = []
        l += [Block(o1, in_channels=1, out_channels=32, kernel=(5, 5), stride=1, padding=0, avg_kernel=2)]
        l += [Block(o, in_channels=32, out_channels=64, kernel=(5, 5), stride=1, padding=0, avg_kernel=2)]
        l += [nn.Flatten()]
        l += [QLinear(o, 1024, 512, bias=False, norm=False)]
        l += [nn.BatchNorm1d(num_features=512)]
        l += [QReLU(o.QReLU_A, channels=512)]
        l += [QLinear(o1, 512, 10, bias=False, norm=False)]
        l += [ScaleBias(channels=10, bias=True)]
        super().__init__(*l)
        # adjusting for clipped activation function so that most of the standartized distributino is in the range [0,1]
        init_net(self, 0.5, 0.5)

    def loss(self, x, y):
        return nn.CrossEntropyLoss(reduction='none')(x, y)

    def loss_network(self, targets):
        return ESequential(self, ELoss(targets, nn.CrossEntropyLoss(reduction='none').forward))

    def net_loss(self, input, targets, method):
        return self.loss_network(targets).forward(input, method=method)


class LeNetTiny(EClassificationNet):
    def __init__(self, o):
        o1 = copy.deepcopy(o)
        o1.QReLU_W = o1.QReLU_W.replace(K=16)  # weights in first and last layers quantized to 16 states = 4 bits
        l = []
        l += [Block(o1, in_channels=1, out_channels=8, kernel=(5, 5), stride=1, padding=0, avg_kernel=2)]
        l += [Block(o, in_channels=8, out_channels=16, kernel=(5, 5), stride=1, padding=0, avg_kernel=2)]
        l += [nn.Flatten()]
        l += [QLinear(o, 256, 128, bias=False, norm=False)]
        l += [nn.BatchNorm1d(num_features=128)]
        l += [QReLU(o.QReLU_A, channels=128)]
        l += [QLinear(o1, 128, 10, bias=False, norm=False)]
        l += [ScaleBias(channels=10, bias=True)]
        super().__init__(*l)
        # adjusting for clipped activation function so that most of the standartized distributino is in the range [0,1]
        init_net(self, 0.5, 0.5)

class LeNetQNN(ESequential):
    def __init__(self, o):
        o1 = copy.deepcopy(o)
        o1.QReLU_W = o1.QReLU_W.replace(K=16)  # weights in first and last layers quantized to 16 states = 4 bits
        l = []
        l += [Block(o1, in_channels=1, out_channels=64,
                    kernel=(5, 5), stride=1, padding=0, avg_kernel=2)]
        l += [Block(o, in_channels=64, out_channels=64,
                    kernel=(5, 5), stride=1, padding=0, avg_kernel=2)]
        l += [nn.Flatten()]
        l += [QLinear(o, 1024, 1024, bias=False, norm=False)]
        l += [nn.BatchNorm1d(num_features=1024)]
        l += [QReLU(o.QReLU_A, channels=1024)]
        l += [QLinear(o1, 1024, 10, bias=False, norm=False)]
        l += [ScaleBias(channels=10, bias=True)]
        super().__init__(*l)
        # adjusting for clipped activation function so that most of the standartized distributino is in the range [0,1]
        init_net(self, 0.5, 0.5)


class Test(EClassificationNet):
    def __init__(self, o):
        l = []
        l += [nn.Flatten()]
        l += [QLinear(o, 28*28, 200, bias=False, norm=False)]
        l += [nn.BatchNorm1d(num_features=200)]
        l += [QReLU(o.QReLU_A, channels=200)]
        l += [QLinear(o, 200, 200, bias=False, norm=False)]
        l += [nn.BatchNorm1d(num_features=200)]
        l += [QReLU(o.QReLU_A, channels=200)]
        l += [QLinear(o, 200, 10, bias=False, norm=False)]
        l += [ScaleBias(channels=10, bias=True)]
        super().__init__(*l)
        # adjusting for clipped activation function so that most of the standartized distributino is in the range [0,1]
        init_net(self, 0.5, 0.5)


# class Test(EClassificationNet):
#     def __init__(self, o):
#         l = []
#         l += [nn.Flatten()]
#         l += [QReLU(o, K=o.A)]
#         l += [QLinear(o, 28*28, 10, bias=False, norm=False)]
#         l += [ScaleBias(in_channels=10, bias=True)]
#         super().__init__(*l)
#         # adjusting for clipped activation function so that most of the standartized distributino is in the range [0,1]
#         init_net(self, 0.5, 0.5)



def create_net(o, print_info=False):
    # using the seed for network initialization
    torch.manual_seed(o.seed)
    if o.net_name == '' or o.net_name == 'LeNetRQ':
        net = LeNetRQ(o).to(dev)
    elif o.net_name == 'LeNetQNN':
        net = LeNetQNN(o).to(dev)
    elif o.net_name == 'LeNetTiny':
        net = LeNetTiny(o).to(dev)
    elif o.net_name == 'Test':
        net = Test(o).to(dev)
    else:
        raise ValueError(f'Unknown network {o.net_name}')
    # net = AllCNN(o).to(dev)
    print(net.__class__.__name__)
    return net
