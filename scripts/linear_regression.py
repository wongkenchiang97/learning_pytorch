import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#prepare regression dataset
X_numpy,y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0],1)

X_test = torch.tensor([5.0],dtype=torch.float32)

#model construct
n_samples, n_features = X.shape
print(f'n_samples, n_features: {n_samples},{n_features}')
input_size = n_features
output_size = 1
model = nn.Linear(input_size,output_size)

# class LinearRegression(nn.Module):
#     def __init__(self,input_dim,output_dim):
#         super(LinearRegression,self).__init__()
#         #define layers
#         self.lin = nn.Linear(input_dim,output_dim)

#     def forward(self,input):
#         return self.lin(input)

# model = LinearRegression(input_size,output_size)
print(f'prediction before training: f(5)={model(X_test).item():.3f}')


#loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

#training
n_iters = 100
for epoch in range(n_iters):
    #predict forward pass
    y_pred = model(X)

    #loss
    l = criterion(Y,y_pred)

    #gradient backward pass
    l.backward()

    #update weights
    optimizer.step()

    #zero gradient
    optimizer.zero_grad()

    #verbose
    [w,b] = model.parameters()
    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

# #testing
print(f'predictoin after training: f(5)={model(X_test).item():.3f}')

#plot 
predicted = model(X).detach().numpy()
plt.plot(X_numpy,y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')
plt.show()