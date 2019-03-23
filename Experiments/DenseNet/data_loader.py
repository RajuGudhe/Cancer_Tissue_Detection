import os
import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_valid_loader(data_dir,
                           name,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    
    error_msg1 = "[!] valid_size should be in the range [0, 1]."
    error_msg2 = "[!] Invalid dataset name."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg1
    assert name in ['cifar10', 'cifar100', 'pcam'], error_msg2

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize
        ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    if name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, 
                    download=True, transform=train_transform)

        valid_dataset = datasets.CIFAR10(root=data_dir, train=True, 
                    download=True, transform=valid_transform)

    elif name == 'pcam':
        data_dir = 'Pcam-data'
        train_dataset = datasets.ImageFolder(os.path.join(data_dir,train), transform=train_transform)

        valid_dataset = datasets.ImageFolder(os.path.join(data_dir, valid), transform=valid_transform)
    
    else:
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, 
            download=True, transform=train_transform)

        valid_dataset = datasets.CIFAR100(root=data_dir, train=True, 
            download=True, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=batch_size, sampler=train_sampler, 
                    num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                    batch_size=batch_size, sampler=valid_sampler, 
                    num_workers=num_workers, pin_memory=pin_memory)


    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=9, 
                                                    shuffle=shuffle, 
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory)
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels, name)

    return (train_loader, valid_loader)

def get_test_loader(data_dir,
                    name,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # define transform
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        normalize
    ])

    if name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir, 
                                   train=False, 
                                   download=True,
                                   transform=transform)
    elif name == 'pcam':
        data_dir = 'PCam-data'
        dataset = datasets.ImageFolder(os.path.join(data_dir, test), transform= transform)
    
    else:
        dataset = datasets.CIFAR100(root=data_dir, 
                                    train=False, 
                                    download=True,
                                    transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return data_loader
