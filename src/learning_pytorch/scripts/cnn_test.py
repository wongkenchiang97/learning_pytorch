import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
num_classes = 10
num_epochs = 20
batch_size = 4
learning_rate = 0.001

#dataset has PILImage images of range [0,1]
#we transform them to Tensors of normalized range [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True,
    transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data',train=False,
    transform=transforms.ToTensor(),download=False)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,
    shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,
    shuffle=False)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet,self).__init__()
        self.conv = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# construct model
model = ConvNet().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

#training loop
n_total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        #original shape: [4, 3, 32, 32] = 4, 3, 1024
        #input_layer: 3 input channels, 6 output channel, 5 kernel size 
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs,labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_step}, loss {loss.item():.4f}')

print('Finished training.')

# testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value,idx
        _,prediction = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct += (prediction == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = prediction[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0*n_correct/n_samples
    print(f'accuracy of network= {acc}%')

    for i in range(10):
        acc  = 100*n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc}%')