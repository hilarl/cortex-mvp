# Imports
from collections import OrderedDict
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
Simple CNN model'.
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


"""
Train the model on the training set.
"""


def train(model, train_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


"""
Evaluate the model on the test set.
"""


def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            loss += criterion(outputs, labels.to(DEVICE)).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(test_loader.dataset), correct / total


"""
Load and preprocess CIFAR-10 (training and test sets).
"""


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


"""
Flower client definition.
"""


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [p.cpu().numpy() for _, p in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, train_loader, epochs=1)
        return self.get_parameters(config), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
