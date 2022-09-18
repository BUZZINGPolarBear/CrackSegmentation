from torch.utils.data import DataLoader
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
# from torch.utils.data.distributed import DistributedSampler
from .dataset import CrackDataSet
from typing import List, Tuple
import os


def initialize_data_loader(batch_size, num_workers) -> Tuple[DataLoader, DataLoader]:
    current_path = os.getcwd();
    train_dataset = CrackDataSet(current_path, split='train')
    valid_dataset = CrackDataSet(current_path, split='val')

    # trainset = torchvision.datasets.CIFAR10(root='path', train=True, download=True, transform=transform)
    # validset = torchvision.datasets.CIFAR10(root='path', train=False, download=False, transform=transform)

    train_sampler = ElasticDistributedSampler(train_dataset)
    # train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(valid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    return train_loader, val_loader