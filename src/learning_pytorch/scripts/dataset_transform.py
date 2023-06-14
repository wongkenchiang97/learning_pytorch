import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class WineDataset(Dataset):

    def __init__(self,transform=None):
        xy = np.loadtxt('wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x = xy[:,1:]
        self.y = xy[:,[0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self,index):
        #dataset[index]
        sample =  self.x[index],self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples

class toTensor:

    def __call__(self,sample):
        input,targets = sample
        return torch.from_numpy(input),torch.from_numpy(targets)

class MulTransform:

    def __init__(self,factor):
        self.factor = factor

    def __call__(self,sample):
        input,output = sample
        input *= self.factor
        return input,output


dataset = WineDataset(transform=None)
first_data = dataset[0]
feature,label = first_data
print(type(feature),type(label))
print(feature)

composed = torchvision.transforms.Compose([toTensor(),MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
feature,label = first_data
print(type(feature),type(label))
print(feature)


