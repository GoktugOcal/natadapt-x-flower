import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import random
import matplotlib.pyplot as plt

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

np_dataset = np.load("./data/Cifar10/train/1.npz")
# print(np_dataset["data"])

# print(np_dataset["data"][0])

# print(torch.Tensor(np_dataset["data"]))

np.load = np_load_old









# dataset_path = "./data"

# train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,
#     transform=transforms.Compose([
#         transforms.RandomCrop(32, padding=4), 
#         transforms.Resize(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     ]))

# targets = train_dataset.targets
# random.shuffle(targets)

# client_1 = targets[:int(len(targets)/2)]
# client_2 = targets[int(len(targets)/2):]


# unique_1, counts_1 = np.unique(client_1, return_counts=True)
# unique_2, counts_2 = np.unique(client_2, return_counts=True)

# width = 0.2
# plt.bar(unique_1 - 0.1, counts_1, width)
# plt.bar(unique_2 + 0.1, counts_2, width)
# plt.show()