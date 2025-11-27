import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
N = 20000
epochs = 50

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.layers(x)

def generate_tensors():
    x = np.random.uniform(-10, 10, N)
    y = np.random.uniform(-10, 10, N)
    val = np.sin(x + 2 * y) * np.exp(-(2 * x + y) ** 2)
    sign = np.vstack([x, y]).T
    val = val.reshape(N, 1)
    sign_tensor = torch.tensor(sign, dtype=torch.float32)
    val_tensor = torch.tensor(val, dtype=torch.float32)
    return sign_tensor, val_tensor

def get_dataset(sign_tensor, val_tensor):
    dataset = torch.utils.data.TensorDataset(sign_tensor, val_tensor)
    n_train = int(N * 0.7)
    n_val = int(N * 0.15)
    n_test = N - n_train - n_val
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128)
    return train_loader, val_loader, test_loader, n_train, n_val, n_test

def work_with_model(model, train_loader, val_loader, test_loader, n_train, n_val, n_test):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        print("Epoch ", epoch+1, ": Train_loss: ", train_loss/n_train, "Val_loss: ", val_loss/n_val)

    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
    print("\nMSE на тесте:", test_loss / n_test)


def vis(model):
    line = torch.linspace(-10, 10, 200)
    with torch.no_grad():
        true_line = np.sin(line.numpy() + 2 * line.numpy()) * np.exp(-(2 * line.numpy() + line.numpy()) ** 2)
        pred_line = model(torch.stack([line, line], dim=1)).numpy().flatten()
    plt.figure(figsize=(10, 6))
    plt.plot(line.numpy(), true_line, label='Исходная функция', linewidth=2)
    plt.plot(line.numpy(), pred_line, label='Модель', linewidth=2)
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    model = Model()
    sign_tensor, val_tensor = generate_tensors()
    train_loader, val_loader, test_loader, n_train, n_val, n_test = get_dataset(sign_tensor, val_tensor)
    work_with_model(model, train_loader, val_loader, test_loader, n_train, n_val, n_test)
    vis(model)

main()
