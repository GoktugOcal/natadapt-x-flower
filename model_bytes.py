import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler
import torchvision.models as models

import io
import sys
import pickle

from copy import deepcopy

model = torch.load("./models/alexnet/model.pth.tar")
# model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# model = models.alexnet(pretrained=True)
model.classifier[1] = nn.Linear(9216,4096)
model.classifier[4] = nn.Linear(4096,1024)
model.classifier[6] = nn.Linear(1024,10)



prim_model = deepcopy(model)

print(">>>>>>>> PLAIN MODEL")
print(type(model))


# ### FILE READ WRITE
# with open("./models/alexnet/model.pth.tar", 'rb') as content_file:
#     some_bytes = content_file.read()
# with open("./temp_models/my_alexnet.pth", "wb") as binary_file:
#     binary_file.write(some_bytes)
# model = torch.load("./temp_models/my_alexnet.pth")

# ### BUFFER 
# buffer = io.BytesIO()
# torch.save(model.state_dict(), buffer)
# model_bytes = buffer.getvalue()
# print(">>>>>>>> BUFFER")
# print(sys.getsizeof(model_bytes))
# print(type(model_bytes))

# ### BUFFER 
# model_buffer = io.BytesIO(model_bytes) 
# model = torch.load(model_buffer)
# print(type(model))
# model.eval()
# print(type(model))


# # Create an instance of the custom AlexNet model
# pretrained_model = CustomAlexNet()


# # Serialize the model using TorchScript
# scripted_model = torch.jit.script(pretrained_model)

# # Convert the serialized model to bytes
# serialized_bytes_io = BytesIO()
# # scripted_model.save(serialized_bytes_io)
# torch.jit.save(scripted_model, serialized_bytes_io)
# serialized_bytes = serialized_bytes_io.getvalue()
# serialized_bytes_io.close()

# received_buffer = BytesIO(serialized_bytes)
# received_scripted_model = torch.jit.load(received_buffer)

# received_scripted_model = received_scripted_model.cuda()


################################


### TORCHSCRIPT
scripted_model = torch.jit.script(model)
print(scripted_model)

buffer = io.BytesIO()
torch.jit.save(scripted_model, buffer)
model_bytes = buffer.getvalue()
buffer.close()

buffer = io.BytesIO(model_bytes)
deserialized_model = torch.jit.load(buffer)
# buffer.close()

model = deserialized_model
print(model.eval())

### FINE TUNING
_NUM_CLASSES = 10
batch_size = 128
num_workers = 4
momentum = 0.9
weight_decay = 1e-4
finetune_lr = 0.001
iterations = 500
print_frequency = 10

train_dataset = datasets.CIFAR10(root="data/", train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=True, shuffle=True)#, sampler=train_sampler)
        
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), finetune_lr, 
                                momentum=momentum, weight_decay=weight_decay)
        
dataloader_iter = iter(train_loader)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# epochs = 10
# for epoch in range(epochs):
#     model.train()
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         labels.unsqueeze_(1)
#         target_onehot = torch.FloatTensor(labels.shape[0], _NUM_CLASSES)
#         target_onehot = target_onehot.to(device)
#         target_onehot.zero_()
#         target_onehot.scatter_(1, labels, 1)
#         labels.squeeze_(1)
#         labels = labels.cuda()
#         target_onehot = target_onehot.cuda()

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, target_onehot)
#         loss.backward()
#         optimizer.step()

#     # Validation
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     print(f"Epoch [{epoch + 1}/{epochs}], Test Accuracy: {accuracy:.2f}%")

# model = model.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
model.train()

for epoch in range(10):
    losses = []
    for inputs, labels in train_loader:

        # Data prep.
        inputs = inputs.to(device)
        labels = torch.nn.functional.one_hot(labels, num_classes=_NUM_CLASSES)
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(device)

        # Forward pass.
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs = outputs.to(device)

        # Compute loss.
        print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        # Backward pass.
        optimizer.zero_grad()
        loss.backward()

        # Update parameters.
        optimizer.step()

    print(f"Epoch {epoch + 1}: Average loss: {sum(losses) / len(losses)}")

# for i in range(iterations):
#     try:
#         (input, target) = next(dataloader_iter)
#     except:
#         dataloader_iter = iter(train_loader)
#         (input, target) = next(dataloader_iter)
        
#     if i % print_frequency == 0:
#         print('Fine-tuning iteration {}'.format(i))
#         sys.stdout.flush()
    
#     target.unsqueeze_(1)
#     target_onehot = torch.FloatTensor(target.shape[0], _NUM_CLASSES)
#     target_onehot.zero_()
#     target_onehot.scatter_(1, target, 1)
#     target.squeeze_(1)
#     input, target = input.cuda(), target.cuda()
#     target_onehot = target_onehot.cuda()

#     pred = model(input)
#     loss = criterion(pred, target_onehot)
#     optimizer.zero_grad()
#     loss.backward()  # compute gradient and do SGD step
#     optimizer.step()




# train_data = torch.randn(100, 10)
# train_target = torch.randint(0, 5, (100,))

# # Step 2c: Train the received model using data on the client side
# optimizer = torch.optim.SGD(deserialized_model.parameters(), lr=0.01)
# criterion = nn.CrossEntropyLoss()

# for epoch in range(100):
#     optimizer.zero_grad()
#     output = deserialized_model(train_data)
#     loss = criterion(output, train_target)
#     loss.backward()
#     optimizer.step()

# print(type(model_bytes))
# print(sys.getsizeof(model_bytes))

# print(">>>>>>>> LOADED MODEL")

# print(loaded_model)
# print(type(loaded_model))
