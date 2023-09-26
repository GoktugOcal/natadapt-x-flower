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

from time import time

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


def test(net, testloader):
    _NUM_CLASSES = 10
    """Validate the model on the test set."""
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, loss = 0, 0.0

    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader):

            # s = time()
            labels.unsqueeze_(1)
            target_onehot = torch.FloatTensor(labels.shape[0], _NUM_CLASSES)
            target_onehot.zero_()
            target_onehot.scatter_(1, labels, 1)
            labels.squeeze_(1)
            labels = labels.to(DEVICE)
            # print("1:", time() - s)

            # s = time()
            outputs = net(images.to(DEVICE))
            # outputs = net(images)
            labels_one_hot = target_onehot.to(DEVICE)
            # print("2:", time() - s)

            # s = time()
            loss += criterion(outputs, labels_one_hot).item()
            # print("3:", time() - s)

            # s = time()            
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            # print("4:", time() - s)

            print(correct)
            
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy





if __name__ == "__main__":

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load("models/alexnet/model_pt21.pth.tar")

    dataset_path = "data/"
    trainloader, testloader = load_data(dataset_path)

    loss, accuracy = test(model, testloader)

    print(accuracy)







