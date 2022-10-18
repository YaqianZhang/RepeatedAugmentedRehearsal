import torch
from models.resnet import ResNet18,Reduced_ResNet18,SupConResNet,SupConResNet_normal
from models.pretrained import ResNet18_pretrained
from torchvision import transforms
import torch.nn as nn


default_trick = {'labels_trick': False, 'kd_trick': False, 'separated_softmax': False,
                 'review_trick': False, 'nmc_trick': False}


input_size_match = {
    'cifar100': [3, 32, 32],
    'cifar10': [3, 32, 32],
    'core50': [3, 128, 128],
'clrs25': [3, 128,128],#[3, 256, 256],
    'mini_imagenet': [3, 84, 84],
    'openloris': [3, 50, 50]
}


n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'core50': 50,
    'clrs25':25,
    'mini_imagenet': 100,
    'openloris': 69
}


transforms_match = {
    'core50': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'clrs25': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor()]),
    'openloris': transforms.Compose([
            transforms.ToTensor()])
}


def setup_architecture(params):
    nclass = n_classes[params.data]

    if params.agent in ['SCR','SCR_META', 'SCP',"SCR_RL_ratio","SCR_RL_iter"]:
        if params.data == 'mini_imagenet':
            if(params.resnet_size == "normal"):
                return SupConResNet_normal(2048, head=params.head)
            else:
                return SupConResNet(640, head=params.head)
        if params.data == 'clrs25':

            if(params.resnet_size == "normal"):
                return SupConResNet_normal(8192, head=params.head)
            else:
                return SupConResNet(2560, head=params.head)

        if params.data == 'core50':
            if(params.resnet_size == "normal"):
                return SupConResNet_normal(8192, head=params.head)
            else:
                return SupConResNet(2560, head=params.head)


        if(params.resnet_size == "normal"):
            return SupConResNet_normal(512, head=params.head)
        else:
            return SupConResNet( head=params.head)

        #return SupConResNet(head=params.head)
    if params.agent == 'CNDPM':
        from models.ndpm.ndpm import Ndpm
        return Ndpm(params)
    if params.data == 'cifar100':
        if(params.resnet_size == "normal"):
            return ResNet18(nclass)
        else:
            return Reduced_ResNet18(nclass)
    elif params.data == 'clrs25':
        if(params.resnet_size == "normal"):
            model= ResNet18(nclass)
            model.linear = nn.Linear(8192, nclass, bias=True)
        else:
            model= Reduced_ResNet18(nclass)
            model.linear = nn.Linear(2560, nclass, bias=True)
        return model
    elif params.data == 'cifar10':

        if(params.resnet_size == "normal"):
            return ResNet18(nclass)
        else:
            return Reduced_ResNet18(nclass)

    elif params.data == 'core50':
        if(params.resnet_size == "normal"):
            model= ResNet18(nclass)
            model.linear = nn.Linear(8192, nclass, bias=True)
        else:
            model= Reduced_ResNet18(nclass)
            model.linear = nn.Linear(2560, nclass, bias=True)

        return model
    elif params.data == 'mini_imagenet':
        if(params.resnet_size == "normal"):
            model= ResNet18(nclass)
            model.linear = nn.Linear(2048, nclass, bias=True)
        else:
            model= Reduced_ResNet18(nclass)

            model.linear = nn.Linear(640, nclass, bias=True)
        return model
    elif params.data == 'openloris':
        return Reduced_ResNet18(nclass)
    else:
        raise NotImplementedError("undefined dataset",params.data)


def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim
