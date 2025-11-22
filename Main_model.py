import numpy as np
import torch
from torch import nn

epochs = 50
batch = 128

class DecoderModel(nn.Module):
    def __init__(self, L_in, L_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(L_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, L_out),
        )
    def forward(self, x):
        return self.layers(x)

def load_dataset(path):
    words = []
    noisy_vectors = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            nums = list(map(float, parts[1:]))
            target_vec = np.array([int(c) for c in word], dtype=np.float32)
            noisy_vec = np.array(nums, dtype=np.float32)
            words.append(target_vec)
            noisy_vectors.append(noisy_vec)
    L_in = len(noisy_vectors[0])
    L_out = len(words[0])
    noisy_tensor = torch.tensor(np.array(noisy_vectors), dtype=torch.float32)
    target_tensor = torch.tensor(np.array(words), dtype=torch.float32)
    return noisy_tensor, target_tensor, L_in, L_out

def get_data_loaders(noisy_tensor, target_tensor):
    N = len(noisy_tensor)
    dataset = torch.utils.data.TensorDataset(noisy_tensor, target_tensor)
    n_train = int(N * 0.7)
    n_val = int(N * 0.10)
    n_test = N - n_train - n_val
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch)
    return train_loader, val_loader, test_loader, n_train, n_val, n_test

def work_with_model(model, train_loader, val_loader, test_loader, n_train, n_val, n_test):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for noisy, target in train_loader:
            optimizer.zero_grad()
            out = model(noisy)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * noisy.size(0)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, target in val_loader:
                out = model(noisy)
                loss = criterion(out, target)
                val_loss += loss.item() * noisy.size(0)
        print("Epoch ", epoch+1, ": Train_loss: ", train_loss/n_train, "Val_loss: ", val_loss/n_val)
    test_loss = 0.0
    with torch.no_grad():
        for noisy, target in test_loader:
            out = model(noisy)
            loss = criterion(out, target)
            test_loss += loss.item() * noisy.size(0)
    print("\nL1 на тесте:", test_loss / n_test)

def main():
    noisy_tensor, target_tensor, L_in, L_out = load_dataset("dataset.txt")
    model = DecoderModel(L_in, L_out)
    train_loader, val_loader, test_loader, n_train, n_val, n_test = get_data_loaders(noisy_tensor, target_tensor)
    work_with_model(model, train_loader, val_loader, test_loader, n_train, n_val, n_test)

main()
