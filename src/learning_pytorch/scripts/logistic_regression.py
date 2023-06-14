import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# prepare dataset
bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target

n_sample,n_feature = X.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

#scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

#construct model: f=sigmoid(wx)
class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_feature,1)

    def forward(self,X):
        y_predicted = torch.sigmoid(self.linear(X))
        return y_predicted

model = LogisticRegression(n_feature)

#construct loss and optimizer
criterion = nn.BCELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

#training
num_iter = 100
for epoch in range(num_iter):
    # forward pass
    y_predicted = model(X_train)
    loss = criterion(y_predicted,y_train)

    #gradient
    loss.backward()

    #update
    optimizer.step()

    #zero grad
    optimizer.zero_grad()

    #verbose
    if (num_iter%10) == 0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')

#evaluation
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum()/ float(y_test.shape[0])
    print(f'accuracy: {acc:.4f}')