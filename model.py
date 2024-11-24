import torch
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Гиперпараметры
input_size = 784
hidden_size = 500
num_classes = 10

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
model.load_state_dict(torch.load('model.ckpt', weights_only=False))
model.eval()

def predict(image):
    image = torch.from_numpy(image)
    with torch.no_grad():
        image = image.reshape(-1, 28*28)
        output = model(image)

        _, predictions = torch.max(output.data, 1)
        return predictions.item()

