import time
import random
from PIL import Image
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import usps
import torch.nn as nn
from data_list import ImageValueList
from arguments import opt

def reform_Ctrain(train_set):
    sel_dat, sel_lab = [], []
    for i in range(len(train_set)):
        if train_set.targets[i] <= 5:
            sel_dat.append(train_set.data[i])
            sel_lab.append(train_set.targets[i])
        elif train_set.targets[i] >= 7:
            sel_dat.append(train_set.data[i])
            sel_lab.append(train_set.targets[i]-1)
        #else:
            #print('6')
    train_set.targets = np.array(sel_lab)
    train_set.data = np.array(sel_dat)
    return train_set

def reform_Strain(train_set):
    sel_dat, sel_lab = [], []
    for i in range(len(train_set)):
        if train_set.labels[i] == 1:
            sel_dat.append(train_set.data[i])
            sel_lab.append(2)
        elif train_set.labels[i] == 2:
            sel_dat.append(train_set.data[i])
            sel_lab.append(1)
        elif train_set.labels[i] >= 8:
            sel_dat.append(train_set.data[i])
            sel_lab.append(train_set.labels[i]-1)
        elif train_set.labels[i] != 7:
            sel_dat.append(train_set.data[i])
            sel_lab.append(train_set.labels[i])
        #else:
            #print(train_set.labels[i])
    train_set.labels = np.array(sel_lab)
    train_set.data = np.array(sel_dat)
    return train_set


def reform_office_subset(input_set):
    sel_dat, sel_lab = [], []
    for i in range(len(input_set)):
        if input_set.values[i] <= 15:
            sel_dat.append(input_set.imgs[i])
            sel_lab.append(input_set.values[i])
    input_set.imgs = np.array(sel_dat)
    input_set.values = np.array(sel_lab)
    return input_set

def get_testloader(task, batch_size):
    if task == 'm2s':
        dataloader = DataLoader(
            datasets.SVHN('./data/SVHN', split='test', download=True,
                        transform=transforms.Compose([
                            transforms.Resize((28,28)),
                            transforms.Grayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
            batch_size=batch_size, shuffle=False
        )
    elif task == 'm2u':
        dataloader = DataLoader(
            usps.USPS('./data/USPS', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.Resize((28,28)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])),
            batch_size=batch_size, shuffle=False
        )
    elif task == 's2m':
        dataloader = DataLoader(
            datasets.MNIST('./data/',train=False, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32,32)),
                       transforms.Lambda(lambda x: x.convert("RGB")),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                    ])),
            batch_size=batch_size, shuffle=False
        )
        # dataloader = DataLoader(
        #     datasets.MNIST('./data/',train=False, download=True,
        #            transform=transforms.Compose([
        #                transforms.Resize((28,28)),
        #                transforms.Grayscale(),
        #                transforms.ToTensor(),
        #                transforms.Normalize((0.5,), (0.5,))
        #             ])),
        #     batch_size=batch_size, shuffle=False
        # )

    elif task == 's2u':
        dataloader = DataLoader(
            usps.USPS('./data/USPS', train=False, download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                    ])),
            batch_size=batch_size, shuffle=False
        )
    elif task == 'u2m':
        dataloader = DataLoader(
            dataset = datasets.MNIST('./data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((16,16)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
            batch_size=batch_size, shuffle=False
        )
        # dataloader = DataLoader(
        #     dataset = datasets.MNIST('./data/',train=False, download=True,
        #            transform=transforms.Compose([
        #                transforms.Resize((28,28)),
        #                transforms.ToTensor(),
        #                transforms.Normalize((0.5,), (0.5,))
        #            ])),
        #     batch_size=batch_size, shuffle=False
        # )
    elif task == 'u2s':
        dataloader = DataLoader(
            dataset = datasets.SVHN('./data/SVHN', split='test', download=True,
                    transform=transforms.Compose([
                        transforms.Resize((16,16)),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])),
            batch_size=batch_size, shuffle=False
        )
        # dataloader = DataLoader(
        #     dataset = datasets.SVHN('./data/SVHN', split='test', download=True,
        #             transform=transforms.Compose([
        #                 transforms.Resize((28,28)),
        #                 transforms.Grayscale(),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize((0.5,), (0.5,))
        #             ])),
        #     batch_size=batch_size, shuffle=False
        # )
    elif task == 'c2s':
        dataloader = DataLoader(
            dataset = reform_Strain(datasets.STL10('./data/STL', split='test', download=True,
                        transform=transforms.Compose([
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))),
            batch_size=batch_size, shuffle=False
        )
    elif task == 's2c':
        dataloader = DataLoader(
            dataset = reform_Ctrain(datasets.CIFAR10('./data/CIFAR', train=False, download=True, 
                        transform=transforms.Compose([ 
                            transforms.Resize((96,96)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))),
            batch_size=batch_size, shuffle=False
        )

    return dataloader


def default_loader(path):
        return Image.open(path).convert('L')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,txt, transform=None,target_transform=None, loader=default_loader):
        super(MyDataset,self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader 
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.imgs)


def get_trainloader(dataset='mnist', batch_size=32, data_size=32, resize_size=256, data_channel=1, da_type='CDA'):
    if dataset=='mnist':
        if data_channel==1:
            dataloader = DataLoader(
                dataset=datasets.MNIST(root='./data/', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomCrop(data_size, padding=4),
                                   transforms.Grayscale(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                               ])),
                batch_size=batch_size, shuffle=True
            )
        elif data_channel==3:
            dataloader = DataLoader(
                dataset=datasets.MNIST(root='./data/', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize((data_size, data_size)),
                                           transforms.Lambda(lambda x: x.convert("RGB")),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ])),
                batch_size=batch_size, shuffle=True
            )
    elif dataset == 'usps':
        if data_channel==1:
            dataloader = DataLoader(
                usps.USPS('./data/USPS', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.RandomCrop(data_size, padding=4),
                              transforms.Grayscale(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ])),
                batch_size=batch_size, shuffle=True
            )
        elif data_channel==3:
            dataloader = DataLoader(
                usps.USPS('./data/USPS', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Resize((data_size, data_size)),
                              transforms.Lambda(lambda x: x.convert("RGB")),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ])),
                batch_size=batch_size, shuffle=True
            )
    elif dataset == 'svhn':
        if data_channel==1:
            dataloader = DataLoader(
                datasets.SVHN('./data/SVHN', split='train', download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(data_size, padding=4),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ])),
                batch_size=batch_size, shuffle=True
            )
        elif data_channel==3:
            dataloader = DataLoader(
                datasets.SVHN('./data/SVHN', split='train', download=True,
                              transform=transforms.Compose([
                                  transforms.Resize((data_size, data_size)),
                                  transforms.Lambda(lambda x: x.convert("RGB")),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
                batch_size=batch_size, shuffle=True
            )

    elif dataset == 'cifar':
        dataloader = DataLoader(
            dataset=reform_Ctrain(datasets.CIFAR10('./data/CIFAR', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Resize((resize_size, resize_size)),
                            transforms.RandomCrop(data_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))),
            batch_size=batch_size, shuffle=True
        )

    elif dataset == 'stl':
        dataloader = DataLoader(
            dataset=reform_Strain(datasets.STL10('./data/STL', split='train', download=True,
                        transform=transforms.Compose([
                            transforms.Resize((resize_size, resize_size)),
                            transforms.RandomCrop(data_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))),
            batch_size=batch_size, shuffle=True
        )
    elif dataset == 'amazon' or dataset == 'dslr' or dataset=='webcam':
        dataset =  ImageValueList(open('./data/office/{}_list.txt'.format(dataset)).readlines(),labels=True, \
                    transform=transforms.Compose([
                                transforms.Resize((resize_size, resize_size)),
                                transforms.RandomCrop(data_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))
        if da_type=='ODA':
            dataset = reform_office_subset(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def get_dataset_testloader(dataset='mnist', batch_size=32, data_size=32, resize_size=256, data_channel=1, da_type='CDA'):
    if dataset == 'mnist':
        if data_channel==1:
            dataloader = DataLoader(
                datasets.MNIST(root='./data/', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.CenterCrop(data_size),
                                   transforms.Grayscale(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                               ])),
                batch_size=batch_size, shuffle=False
            )
        elif data_channel==3:
            dataloader = DataLoader(
                datasets.MNIST(root='./data/', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(data_size),
                                   transforms.Lambda(lambda x: x.convert("RGB")),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])),
                batch_size=batch_size, shuffle=False
            )
    elif dataset == 'usps':
        if data_channel == 1:
            dataloader = DataLoader(
                usps.USPS('./data/USPS', train=False, download=True,
                          transform=transforms.Compose([
                              transforms.CenterCrop(data_size),
                              transforms.Grayscale(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ])),
                batch_size=batch_size, shuffle=False
            )
        elif data_channel == 3:
            dataloader = DataLoader(
                usps.USPS('./data/USPS', train=False, download=True,
                          transform=transforms.Compose([
                              transforms.Resize((data_size, data_size)),
                              transforms.Lambda(lambda x: x.convert("RGB")),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ])),
                batch_size=batch_size, shuffle=False
            )
    elif dataset == 'svhn':
        if data_channel==1:
            dataloader = DataLoader(
                datasets.SVHN('./data/SVHN', split='test', download=True,
                            transform=transforms.Compose([
                                transforms.CenterCrop(data_size),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ])),
                batch_size=batch_size, shuffle=False
            )
        elif data_channel==3:
            dataloader = DataLoader(
                datasets.SVHN('./data/SVHN', split='test', download=True,
                            transform=transforms.Compose([
                                transforms.Resize((data_size, data_size)),
                                transforms.Lambda(lambda x: x.convert("RGB")),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])),
                batch_size=batch_size, shuffle=False
            )
    elif dataset == 'cifar':
        dataloader = DataLoader(
            dataset=reform_Ctrain(datasets.CIFAR10('./data/CIFAR', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.Resize(resize_size),
                            transforms.CenterCrop(data_size),
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))),
            batch_size=batch_size, shuffle=False
        )
    elif dataset == 'stl':
        dataloader = DataLoader(
            dataset=reform_Strain(datasets.STL10('./data/STL', split='test', download=True,
                        transform=transforms.Compose([
                            transforms.Resize(resize_size),
                            transforms.CenterCrop(data_size),
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))),
            batch_size=batch_size, shuffle=False
        )
    elif dataset == 'amazon' or dataset == 'dslr' or dataset == 'webcam':
        dataset = ImageValueList(open('./data/office/{}_test_list.txt'.format(dataset)).readlines(), labels=True, \
                               transform=transforms.Compose([
                                   transforms.Resize(resize_size),
                                   transforms.CenterCrop(data_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ]))
        if da_type=='PDA':
            dataset = reform_office_subset(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def create_target_samples_v2(n=1, dataset = 'mnist', clss=10, random=True, data_size=28, data_channel=1, resize_size=256, da_type='CDA'):
    if dataset == 'svhn':
        if data_channel==1:
            dataset = datasets.SVHN('./data/SVHN', split='train', download=True,
                            transform=transforms.Compose([
                                transforms.CenterCrop(data_size),
                                # transforms.RandomCrop(data_size, padding=4),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ]))
        elif data_channel==3:
            dataset = datasets.SVHN('./data/SVHN', split='train', download=True,
                            transform=transforms.Compose([
                                transforms.Resize((data_size, data_size)),
                                transforms.Lambda(lambda x: x.convert("RGB")),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
    elif dataset == 'usps':
        if data_channel == 1:
            dataset = usps.USPS('./data/USPS', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.CenterCrop(data_size),
                        # transforms.RandomCrop(data_size, padding=4),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]))
        elif data_channel == 3:
            dataset = usps.USPS('./data/USPS', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize((data_size,data_size)),
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]))
    elif dataset == 'mnist':
        if data_channel==1:
            dataset = datasets.MNIST('./data/', train='train', download=True,
                       transform=transforms.Compose([
                           transforms.CenterCrop(data_size),
                           # transforms.RandomCrop(data_size, padding=4),
                           transforms.Grayscale(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),(0.5,))
                       ]))
        elif data_channel==3:
            dataset = datasets.MNIST('./data/', train='train', download=True,
                       transform=transforms.Compose([
                           transforms.Resize((data_size,data_size)),
                           transforms.Lambda(lambda x: x.convert("RGB")),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    elif dataset=='stl':
        dataset = datasets.STL10('./data/STL', split='train', download=True,
                        transform=transforms.Compose([
                            transforms.Resize((data_size, data_size)),
                            # transforms.Resize((resize_size, resize_size)),
                            # transforms.RandomCrop(data_size),
                            # transforms.RandomHorizontalFlip(),
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))
        dataset = reform_Strain(dataset)
    elif dataset=='cifar':
        dataset = datasets.CIFAR10('./data/CIFAR', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Resize((data_size, data_size)),
                            # transforms.Resize((resize_size, resize_size)),
                            # transforms.RandomCrop(data_size),
                            # transforms.RandomHorizontalFlip(),
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))
        dataset = reform_Ctrain(dataset)

    elif dataset == 'amazon' or dataset == 'dslr' or dataset=='webcam':
        dataset=ImageValueList(open('./data/office/{}_train_list.txt'.format(dataset)).readlines(),labels=True, \
                    transform=transforms.Compose([
                                transforms.Resize((data_size, data_size)),
                                # transforms.Resize((resize_size, resize_size)),
                                # transforms.RandomCrop(data_size),
                                # transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))
        if da_type=='PDA':
            dataset = reform_office_subset(dataset)
    else:
        print('Warning: Unknown task!')

    X,Y=[],[]
    classes=clss*[n]

    start_idx = [0, 20, 30, 40, 50][opt['seed_id']]

    i=0
    if random==True:
        sample_idx = np.random.randint(0,len(dataset),1000)
    while True:
        if len(X)==n*clss:
            break
        if random==True:
            x, y = dataset[sample_idx[i]]
        else:
            # x,y=dataset[i]
            # x, y = dataset[i%len(dataset)]
            x, y = dataset[(i+start_idx) % len(dataset)]
        if classes[y]>0:
            X.append(x)
            Y.append(y)
            classes[y]-=1
        i+=1

    assert (len(X)==n*clss)
    return torch.stack(X,dim=0),torch.from_numpy(np.array(Y))
