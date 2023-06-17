import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_input_features) -> None:
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)

#train your model

FILE = "src/learning_pytorch/model/model.pth"

# save model
# torch.save(model,FILE)
# torch.save(model.state_dict(),FILE)
for param in model.parameters():
    print(param)

# load model
# model = torch.load(FILE)
# model.eval()
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval() 

# for param in model.parameters():
#     print(param)
print(model.state_dict())
