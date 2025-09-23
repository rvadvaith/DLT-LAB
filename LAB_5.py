import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_activation_functions():
    x = torch.linspace(-10, 10, 100)
    activations = {
        'Linear': x,
        'Sigmoid': torch.sigmoid(x),
        'Tanh': torch.tanh(x),
        'ReLU': F.relu(x),
        'Leaky ReLU': F.leaky_relu(x, negative_slope=0.1),
    }
    plt.figure(figsize=(12, 6))
    for i, (name, y) in enumerate(activations.items()):
        plt.subplot(2, 3, i + 1)
        plt.plot(x.numpy(), y.numpy())
        plt.title(name)
        plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_activation_functions()

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

class NetReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class NetSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return self.fc2(x)

class NetTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

def train_model(model, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
    acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
    print(f'{name} Test Accuracy: {acc:.4f}')

models = {
    "ReLU": NetReLU(),
    "Sigmoid": NetSigmoid(),
    "Tanh": NetTanh()
}

for name, model in models.items():
    train_model(model, name)