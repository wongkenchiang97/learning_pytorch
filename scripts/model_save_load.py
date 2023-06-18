import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_input_features) -> None:
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

# create model
model = Model(n_input_features=6)

#train your model
# learning_rate = 0.001
# optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
# print(optimizer.state_dict())

FILE = "src/learning_pytorch/model/model.pth"

#create checkpoint
# checkpoint = {
#     "epoch": 90,
#     "model_state": model.state_dict(),
#     "optimizer_state": optimizer.state_dict()
# }

#save checkpoint
# torch.save(checkpoint,"src/learning_pytorch/model/checkpoint.pth")

# save model
# torch.save(model,FILE)
# torch.save(model.state_dict(),FILE)
# for param in model.parameters():
#     print(param)

# load model
# model = torch.load(FILE)
# model.eval()
# loaded_model = Model(n_input_features=6)
# loaded_model.load_state_dict(torch.load(FILE))
# loaded_model.eval() 

# for param in model.parameters():
#     print(param)
# print(loaded_model.state_dict())

#load checkpoint
loaded_checkpoint = torch.load("src/learning_pytorch/model/checkpoint.pth")
epoch = loaded_checkpoint["epoch"]
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(),lr=0)
model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])
print(optimizer.state_dict())

""" SAVING ON GPU/CPU 

# 1) Save on GPU, Load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

# 2) Save on GPU, Load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# Note: Be sure to use the .to(torch.device('cuda')) function 
# on all model inputs, too!

# 3) Save on CPU, Load on GPU
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)

# This loads the model to a given GPU device. 
# Next, be sure to call model.to(torch.device('cuda')) to convert the model’s parameter tensors to CUDA tensors
"""
