import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from non_iid_generator.utils.dataset_utils import check, separate_data, split_data, save_file
from utils.customDataset import CustomDataset
from argparse import ArgumentParser

random.seed(1)
np.random.seed(1)
# num_clients = 20
# num_classes = 10
dir_path = "./data/Cifar10/"


# Allocate data to users
def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    dataset_image = []
    dataset_image.extend(trainset.data)
    dataset_image.extend(testset.data)
    dataset_image = np.array(dataset_image)

    train_targets = trainset.targets
    test_targets = testset.targets

    dataset_label = []
    dataset_label.extend(train_targets)
    dataset_label.extend(test_targets)
    dataset_label = np.array(dataset_label)


    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-nc", "--nc", type=int,
                            help="Number of clients.")
    arg_parser.add_argument("-c", "--c", type=int,
                            help="Numer of classes")
    arg_parser.add_argument("-niid", "--niid", type=str,
                            help="Non-IID format.")
    arg_parser.add_argument("-b", "--b", type=str,
                            help="Balanced scenario")
    arg_parser.add_argument("-p", "--p",
                            help="Pathological or Practical(Dirichlet) scenerio . (pat/dir)")

    args = arg_parser.parse_args()

    print(args)

    niid = True if args.niid == "noniid" else False
    balance = True if args.b == "balance" else False
    partition = args.p if args.p != "-" else None

    num_clients = args.nc
    num_classes = args.c

    generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition)