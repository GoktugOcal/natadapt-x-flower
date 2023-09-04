from nets import alexnet
import torch

model = torch.load('models/alexnet/model.pth')
print(model.state_dict())