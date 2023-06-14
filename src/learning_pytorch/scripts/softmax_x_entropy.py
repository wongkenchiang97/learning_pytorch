import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()
Y = torch.tensor([2,0,1])

#nsamples x nclasses = 1x3
Y_hat_good = torch.tensor([[1.0,1.0,3.1],[2.0,1.0,0.1],[1.0,2.0,0.1]])
Y_hat_bad = torch.tensor([[0.5,2.0,0.3],[1.0,2.0,0.1],[2.0,1.0,0.1]])

l1 = loss(Y_hat_good,Y)
l2 = loss(Y_hat_bad,Y)

print(l1.item())
print(l2.item())

#get prediction
_,prediction = torch.max(Y_hat_good,1)
_,prediction2 = torch.max(Y_hat_bad,1)

print(prediction)
print(prediction2)

