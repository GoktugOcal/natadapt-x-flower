import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        print(np.asarray(img).astype(np.float32).shape)
        img = Image.fromarray(np.asarray(img).astype(np.float32)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        sample = {
            'data': img,
            'label': self.labels[idx]
        }
        
        return sample