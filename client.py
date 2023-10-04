from argparse import ArgumentParser
import warnings
from collections import OrderedDict
import json
import io
import os
import sys
import pickle
import base64
from traceback import print_exc

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
from time import sleep, time

from non_iid_generator.customDataset import CustomDataset

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = os.environ["TORCH_DEVICE"]


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
    latency_measurements = []
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            
            # if len(labels.shape) > 1: labels = labels.reshape(len(labels))

            # labels.unsqueeze_(1)
            target_onehot = torch.FloatTensor(labels.shape[0], _NUM_CLASSES)
            target_onehot.zero_()
            target_onehot.scatter_(1, labels, 1)
            # labels.squeeze_(1)
            labels = labels.to(DEVICE)
            
            s = time()
            outputs = net(images.to(DEVICE))
            labels_one_hot = target_onehot.to(DEVICE)
            latency_measurements.append(time() - s)

            loss += criterion(outputs, labels_one_hot).item()
            correct += (torch.max(outputs.data, 1)[1] == labels.squeeze_(1)).sum().item()
            # print(correct)
            # print(f"ACC: {correct / len(outputs.data)}")
    # print(correct, len(testloader.dataset))
    accuracy = correct / len(testloader.dataset)
    print("Test Accuracy :", round(accuracy, 3), "| Total no samples :", len(testloader.dataset))
    return loss, accuracy, latency_measurements

def fine_tune(model, iterations, train_loader, print_frequency=100):
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

    # train_loader, val_loader = load_data(train_dataset_path, test_dataset_path)

    criterion = torch.nn.BCEWithLogitsLoss()
    
    _NUM_CLASSES = 10
    optimizer = torch.optim.SGD(
        model.parameters(),
        finetune_lr, 
        momentum=momentum,
        weight_decay=weight_decay)

    model = model.to(DEVICE)
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
        
        # Ensure the target shape is sth like torch.Size([batch_size])
        if len(labels.shape) > 1: labels = labels.reshape(len(labels))

        target.unsqueeze_(1)
        target_onehot = torch.FloatTensor(target.shape[0], _NUM_CLASSES)
        target_onehot.zero_()
        target_onehot.scatter_(1, target, 1)
        target.squeeze_(1)
        input, target = input.to(DEVICE), target.to(DEVICE)
        target_onehot = target_onehot.to(DEVICE)

        pred = model(input)
        loss = criterion(pred, target_onehot)
        optimizer.zero_grad()
        loss.backward()  # compute gradient and do SGD step
        optimizer.step()

    return model


def load_data(train_dataset_path, test_dataset_path):
    """Load CIFAR-10 (training and test set)."""
    # trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    # testset = CIFAR10("./data", train=False, download=True, transform=trf)
    # return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

    batch_size = 128
    momentum = 0.9
    weight_decay = 1e-4
    finetune_lr = 0.001

    train_data = pickle.load(open(train_dataset_path, "rb"))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True)

    test_data = pickle.load(open(test_dataset_path, "rb"))
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True)

    # dataset_path = "./data/"

    # train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,
    # transform=transforms.Compose([
    #     transforms.RandomCrop(32, padding=4), 
    #     transforms.Resize(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ]))

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size, 
    #     pin_memory=True,
    #     shuffle=True)#, sampler=train_sampler)


    # val_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True,
    # transform=transforms.Compose([
    #     transforms.Resize(224), 
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ]))
    # test_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     pin_memory=True) #, sampler=valid_sampler)
    
    return train_loader, test_loader

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)

# dataset_path = "data/"
# trainloader, testloader = load_data(dataset_path)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, log_file, train_loader, test_loader):
        super().__init__()
        self.client_id = client_id
        self.log_file = log_file

        self.trainLoader, self.testLoader = train_loader, test_loader
        
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
        self.netadapt_config = config

        ## Receive the model
        buffer = io.BytesIO(config["network_arch"])
        self.model = torch.jit.load(buffer)
        buffer.close()
        ## Check the received model
        self.model.eval()
        # print(type(self.model))
        # print(self.model)

        train_dataset_path = f"./data/Cifar10/train/{self.client_id}.pkl"
        test_dataset_path = f"./data/Cifar10/test/{self.client_id}.pkl"

        self.model = fine_tune(self.model, 5, self.trainLoader, print_frequency=1)

        # train(net, trainloader, epochs=1)
        return self.get_parameters(None), len(self.trainLoader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # train_dataset_path = f"./data/Cifar10/train/{self.client_id}.pkl"
        # test_dataset_path = f"./data/Cifar10/test/{self.client_id}.pkl"        
        # trainLoader, testLoader = load_data(train_dataset_path, test_dataset_path)
        
        loss, accuracy, latency_measurements = test(self.model, self.testLoader)
        mean_latency = np.mean(latency_measurements)

        with open(self.log_file, "a") as f:
            line = ",".join(
                [
                    str(self.netadapt_config["netadapt_iteration"]),
                    str(self.netadapt_config["block_id"]),
                    str(self.netadapt_config["server_round"]),
                    str(accuracy),
                    str(loss)
                ]
            )
            line += "\n"
            f.write(line)

        return loss, len(self.testLoader.dataset), {"accuracy": accuracy}


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('working_folder', type=str, 
                            help='Root folder where models, related files and history information are saved.')
    arg_parser.add_argument("-no", "--no", type=int,
                            help="id of the client")
    args = arg_parser.parse_args()
    client_id = args.no

    client_folder_name = os.path.join(args.working_folder, "client" + "_" + str(client_id))
    log_file = os.path.join(client_folder_name, "logs.txt")

    train_dataset_path = f"./data/Cifar10/train/{client_id}.pkl"
    test_dataset_path = f"./data/Cifar10/test/{client_id}.pkl"
    trainloader, testloader = load_data(train_dataset_path, test_dataset_path)

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Iteration,Block,Round,Accuracy,Loss\n")

    while True:
        print("Waiting for server connnection...")
        try:
            # Start Flower client
            fl.client.start_numpy_client(
                server_address="10.0.0.20:8080",
                # server_address="127.0.0.1:8080",
                client=FlowerClient(
                    client_id,
                    log_file,
                    train_loader= trainloader,
                    test_loader=testloader
                ),
                grpc_max_message_length = 1073741824
            )
        except KeyboardInterrupt:
            break
        except Exception as e:
            # logging.Warning(e)
            print_exc()
            sleep(3)