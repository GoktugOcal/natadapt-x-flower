from dotenv import load_dotenv
load_dotenv()

from argparse import ArgumentParser
import os
import time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import pickle
import numpy as np

from copy import deepcopy

from flwr.common.parameter import *
from flwr.server.strategy.aggregate import *
from collections import OrderedDict

import nets as models
import functions as fns
from non_iid_generator.customDataset import CustomDataset
from utils import imagenet_loader

_NUM_CLASSES = 10
# DEVICE = os.environ["TORCH_DEVICE"]
# DEVICE = "cuda"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
   

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def get_avg(self):
        return self.avg
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    
def compute_accuracy(output, target):
    output = output.argmax(dim=1)
    acc = 0.0
    acc = torch.sum(target == output).item()
    acc = acc/output.size(0)*100
    return acc
    

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    
    print('===================================================================')
    end = time.time()
    
    for i, (images, target) in enumerate(train_loader):
        
        # # Ensure the target shape is sth like torch.Size([batch_size])
        if len(target.shape) > 1: target = target.reshape(len(target))

        target.unsqueeze_(1)
        target_onehot = torch.FloatTensor(target.shape[0], _NUM_CLASSES)
        target_onehot.zero_()
        target_onehot.scatter_(1, target, 1)
        target.squeeze_(1)
        
        images = images.to(DEVICE)
        target_onehot = target_onehot.to(DEVICE)
        target = target.to(DEVICE)

        output = model(images)
        loss = criterion(output, target_onehot)
        
        # measure accuracy and record loss
        batch_acc = compute_accuracy(output, target)
        
        losses.update(loss.item(), images.size(0))
        acc.update(batch_acc, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update statistics
        estimated_time_remained = batch_time.get_avg()*(len(train_loader)-i-1)
        fns.update_progress(i, len(train_loader), 
            ESA='{:8.2f}'.format(estimated_time_remained)+'s',
            loss='{:4.2f}'.format(loss.item()),
            acc='{:4.2f}%'.format(float(batch_acc))
            )

    print()
    print('Finish epoch {}: time = {:8.2f}s, loss = {:4.2f}, acc = {:4.2f}%'.format(
            epoch+1, batch_time.get_avg()*len(train_loader), 
            float(losses.get_avg()), float(acc.get_avg())))
    print('===================================================================')
    return


def eval(test_loader, model, args):
    batch_time = AverageMeter()
    acc = AverageMeter()

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(test_loader):

        if len(target.shape) > 1: target = target.reshape(len(target))

        images = images.to(DEVICE)
        target = target.to(DEVICE)
        
        output = model(images)
        batch_acc = compute_accuracy(output, target)
        acc.update(batch_acc, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Update statistics
        estimated_time_remained = batch_time.get_avg()*(len(test_loader)-i-1)
        fns.update_progress(i, len(test_loader), 
            ESA='{:8.2f}'.format(estimated_time_remained)+'s',
            acc='{:4.2f}'.format(float(batch_acc))
            )
    print()
    print('Test accuracy: {:4.2f}% (time = {:8.2f}s)'.format(
            float(acc.get_avg()), batch_time.get_avg()*len(test_loader)))
    print('===================================================================')
    return float(acc.get_avg())
            
def client(global_model, client_id, args):

    model = deepcopy(global_model)

    train_dataset_path = os.path.join(args.data, "train", f"{client_id}.pkl")
    train_data = pickle.load(open(train_dataset_path, "rb"))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True)

    test_dataset_path = os.path.join(args.data, "test", f"{client_id}.pkl")
    test_data = pickle.load(open(test_dataset_path, "rb"))
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=True)

    # Network
    cudnn.benchmark = True
    num_classes = _NUM_CLASSES
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    # Train & evaluation
    best_acc = 0
    for epoch in range(args.epochs):
        print('Epoch [{}/{}]'.format(epoch+1, args.epochs))
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        acc = eval(test_loader, model, args)
        
        if acc > best_acc:
            best_acc = acc
        print(' ')
    print('Best accuracy:', best_acc)
        
    best_acc = eval(test_loader, model, args)
    print('Best accuracy:', best_acc)

    with open(args.logfilename, "a") as f:
        f.write(f"{args.round_no},{client_id},test,{best_acc}\n")

    return model, len(train_data.data)

# def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
#     """Convert parameters object to NumPy ndarrays."""
#     return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]

# def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

def fedavg(weights_results):
    parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
    return parameters_aggregated

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

if __name__ == '__main__':
    # Parse the input arguments.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('data', metavar='DIR', help='path to dataset')
    arg_parser.add_argument('-m', '--model_name', type=str)
    arg_parser.add_argument('-nc', '--no_clients', type=int)
    arg_parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    arg_parser.add_argument('-nr', '--no_rounds', type=int)
    arg_parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run (default: 150)')
    arg_parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='batch size (default: 128)')
    arg_parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (defult: 0.1)', dest='lr')
    arg_parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
    arg_parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
    arg_parser.add_argument('--dir', type=str, default='models/', dest='save_dir', 
                            help='path to save models (default: models/')
    args = arg_parser.parse_args()
    print(args)


    args.logfilename = os.path.join("./logs",args.data.split("/")[-1] + ".txt")
    with open(args.logfilename, "w") as f:
        f.write("RoundNo,ClientNo,Dataset,Accuracy\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4), 
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True,
        transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True,
        transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    global_model = torch.load(args.model_name, map_location=DEVICE)
    global_model = global_model.to(DEVICE)

    for round_no in range(args.no_rounds):
        args.round_no = round_no
        print(f"Round No: {round_no}")

        weights = []

        for client_id in range(args.no_clients):
            print(f"Client No: {client_id}")
            local_model, no_samples = client(global_model, client_id, args)
            weights.append((get_parameters(local_model), no_samples))

        parameters_aggregated = fedavg(weights)
        parameters = parameters_to_ndarrays(parameters_aggregated)
        # Set parameters
        params_dict = zip(global_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        global_model.load_state_dict(state_dict, strict=True)

        global_train_acc = eval(train_loader, global_model, args)
        global_test_acc = eval(test_loader, global_model, args)

        with open(args.logfilename, "a") as f:
            f.write(f"{args.round_no},server,train,{global_train_acc}\n")
            f.write(f"{args.round_no},server,test,{global_test_acc}\n")


    

 

    