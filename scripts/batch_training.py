import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt('wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self,index):
        #dataset[index]
        return self.x[index],self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples

dataset = WineDataset()
data_loader = DataLoader(dataset=dataset,batch_size=4,shuffle=True, num_workers=2)
dataiter = iter(data_loader)
data = next(dataiter)
features,label = data
# print(f'features: {features} \n label:{label}')

#training
num_epoch = 2
total_samples = len(dataset)
n_iter = math.ceil(total_samples/4)
print(total_samples,n_iter)

for epoch in range(num_epoch):
    for i ,(inputs,label) in enumerate(data_loader):
        #forward backwad pass, update
        if (i+1)%5 == 0:
            print(f'epoch {epoch+1},step {i+1}/{n_iter}, inputs {inputs.shape}')


