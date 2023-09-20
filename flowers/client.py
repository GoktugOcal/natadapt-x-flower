import warnings
from collections import OrderedDict
import json
import io
import sys
import pickle
import base64

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from time import sleep


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    _NUM_CLASSES = 10
    """Validate the model on the test set."""
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            
            labels.unsqueeze_(1)
            target_onehot = torch.FloatTensor(labels.shape[0], _NUM_CLASSES)
            target_onehot.zero_()
            target_onehot.scatter_(1, labels, 1)
            labels = labels.to(DEVICE)

            outputs = net(images.to(DEVICE))
            labels_one_hot = target_onehot.to(DEVICE)

            loss += criterion(outputs, labels_one_hot).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            # print(f"ACC: {correct / len(outputs.data)}")
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def fine_tune(model, iterations, dataset_path, print_frequency=100):
    '''
        short-term fine-tune a simplified model
        
        Input:
            `model`: model to be fine-tuned.
            `iterations`: (int) num of short-term fine-tune iterations.
            `print_frequency`: (int) how often to print fine-tune info.
        
        Output:
            `model`: fine-tuned model.
    '''

    # Data loaders for fine tuning and evaluation.
    batch_size = 128
    momentum = 0.9
    weight_decay = 1e-4
    finetune_lr = 0.001

    train_loader, val_loader = load_data(dataset_path)

    criterion = torch.nn.BCEWithLogitsLoss()
    
    _NUM_CLASSES = 10
    optimizer = torch.optim.SGD(
        model.parameters(),
        finetune_lr, 
        momentum=momentum,
        weight_decay=weight_decay)

    model = model.cuda()
    model.train()
    dataloader_iter = iter(train_loader)
    for i in range(iterations):
        try:
            (input, target) = next(dataloader_iter)
        except:
            dataloader_iter = iter(train_loader)
            (input, target) = next(dataloader_iter)
            
        if i % print_frequency == 0:
            print('Fine-tuning iteration {}'.format(i))
            sys.stdout.flush()
        
        target.unsqueeze_(1)
        target_onehot = torch.FloatTensor(target.shape[0], _NUM_CLASSES)
        target_onehot.zero_()
        target_onehot.scatter_(1, target, 1)
        target.squeeze_(1)
        input, target = input.cuda(), target.cuda()
        target_onehot = target_onehot.cuda()

        pred = model(input)
        loss = criterion(pred, target_onehot)
        optimizer.zero_grad()
        loss.backward()  # compute gradient and do SGD step
        optimizer.step()

    return model


def load_data(dataset_path):
    """Load CIFAR-10 (training and test set)."""
    # trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    # testset = CIFAR10("./data", train=False, download=True, transform=trf)
    # return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

    batch_size = 128
    # momentum = 0.9
    # weight_decay = 1e-4
    # finetune_lr = 0.001

    train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, 
        pin_memory=True,
        shuffle=True)#, sampler=train_sampler)


    val_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True) #, sampler=valid_sampler)
    
    return train_loader, val_loader

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)

dataset_path = "data/"
trainloader, testloader = load_data(dataset_path)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        if hasattr(self, "model"):
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        else:
            return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    def get_properties(self, ins):
        print("Server requested for the properties.")
        return self.properties

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("############ FIT ############")

        # if hasattr(self, "model"):
        ## Receive the model
        buffer = io.BytesIO(config["network_arch"])
        self.model = torch.jit.load(buffer)
        buffer.close()
        ## Check the received model
        self.model.eval()
        print(type(self.model))
        print(self.model)

        self.model = fine_tune(self.model, 5, "data/", print_frequency=1)

        # train(net, trainloader, epochs=1)
        return self.get_parameters(None), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# fl.client.start_numpy_client(
#     server_address="10.0.0.20:8080",
#     # server_address="127.0.0.1:8080",
#     client=FlowerClient(),
# )

while True:
    print("retry")
    try:
        # Start Flower client
        fl.client.start_numpy_client(
            server_address="10.0.0.20:8080",
            # server_address="127.0.0.1:8080",
            client=FlowerClient(),
        )
    except KeyboardInterrupt:
        break
    except:
        # pass
        sleep(3)