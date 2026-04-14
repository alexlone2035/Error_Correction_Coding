import torch
from torch import nn
import random
import numpy as np

device = torch.device('cuda')

epochs = 150
batch = 128
L_mid = 128

def init_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class DecoderModel(nn.Module):
    def __init__(self, L_in, L_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(L_in, L_mid),
            nn.ReLU(),
            nn.Linear(L_mid, L_mid),
            nn.ReLU(),
            nn.Linear(L_mid, L_mid),
            nn.ReLU(),
            nn.Linear(L_mid, L_mid),
            nn.ReLU(),
            nn.Linear(L_mid, L_out),
        )
        self.data_mean = 0.0
        self.data_std = 1.0

    def forward(self, x):
        return self.layers(x)

    @classmethod
    def train_model(cls, train_path, test_path, save_path="checkpoint.pth"):

        noisy_train_tensor, target_train_tensor, noisy_test_tensor, target_test_tensor, L_in, L_out, mean, std = load_dataset(train_path, test_path)
        train_loader, val_loader, test_loader, n_train, n_val, n_test = get_data_loaders(noisy_train_tensor, target_train_tensor, noisy_test_tensor, target_test_tensor)

        model = cls(L_in, L_out).to(device)

        model.data_mean = mean
        model.data_std = std

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for noisy, target in train_loader:
                noisy = noisy.to(device)
                target = target.to(device)
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
                    noisy = noisy.to(device)
                    target = target.to(device)
                    out = model(noisy)
                    loss = criterion(out, target)
                    val_loss += loss.item() * noisy.size(0)
            print("Epoch ", epoch + 1, ": Train_loss: ", train_loss / n_train, "Val_loss: ", val_loss / n_val)

        test_loss = 0.0
        with torch.no_grad():
            for noisy, target in test_loader:
                noisy = noisy.to(device)
                target = target.to(device)
                out = model(noisy)
                loss = criterion(out, target)
                test_loss += loss.item() * noisy.size(0)
        print("\nBCEWithLogitsLoss на тесте:", test_loss / n_test)

        checkpoint = {
            'L_in': L_in,
            'L_out': L_out,
            'mean': mean,
            'std': std,
            'state_dict': model.state_dict()
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint сохранен: {save_path}")

        return model

    @classmethod
    def load_from_checkpoint(cls, load_path="checkpoint.pth"):
        checkpoint = torch.load(load_path, map_location=device)
        model = cls(checkpoint['L_in'], checkpoint['L_out'])
        model.load_state_dict(checkpoint['state_dict'])
        model.data_mean = checkpoint['mean']
        model.data_std = checkpoint['std']
        model.to(device)
        model.eval()
        return model

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = (x - self.data_mean) / self.data_std
            x = x.to(device)
            prediction = self(x)
        return prediction

def padding(train_noisy_vectors, train_words, test_noisy_vectors, test_words):
    max_word = max(len(word) for word in train_words)
    max_word = max(max_word, max(len(word) for word in test_words))
    max_vector = max(len(vector) for vector in train_noisy_vectors)
    max_vector = max(max_vector, max(len(vector) for vector in test_noisy_vectors))
    padded_train_words = []
    padded_train_vectors = []
    padded_test_words = []
    padded_test_vectors = []
    for word in train_words:
        if len(word) < max_word:
            pad_word = [0] * (max_word - len(word)) + word
        else:
            pad_word = word
        padded_train_words.append(pad_word)
    for word in test_words:
        if len(word) < max_word:
            pad_word = [0] * (max_word - len(word)) + word
        else:
            pad_word = word
        padded_test_words.append(pad_word)
    for vector in train_noisy_vectors:
        if len(vector) < max_vector:
            pad_vector = [0] * (max_vector - len(vector)) + vector
        else:
            pad_vector = vector
        padded_train_vectors.append(pad_vector)
    for vector in test_noisy_vectors:
        if len(vector) < max_vector:
            pad_vector = [0] * (max_vector - len(vector)) + vector
        else:
            pad_vector = vector
        padded_test_vectors.append(pad_vector)
    return padded_train_words, padded_train_vectors, padded_test_words, padded_test_vectors

def normalization(padded_train_words, padded_train_vectors, padded_test_words, padded_test_vectors):
    X_train = np.array(padded_train_vectors, dtype=np.float32)
    Y_train = np.array(padded_train_words, dtype=np.float32)
    X_test = np.array(padded_test_vectors, dtype=np.float32)
    Y_test = np.array(padded_test_words, dtype=np.float32)
    mean = X_train.mean()
    std = X_train.std()
    if std == 0: std = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    L_in = X_train.shape[1]
    L_out = Y_train.shape[1]
    return torch.tensor(X_train), torch.tensor(Y_train), torch.tensor(X_test), torch.tensor(Y_test), L_in, L_out, mean, std

def load_dataset(train_path, path):
    train_words = []
    train_noisy_vectors = []
    test_words = []
    test_noisy_vectors = []
    with open(train_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            nums = list(map(float, parts[1:]))
            target_vec = list(int(c) for c in word)
            noisy_vec = nums
            train_words.append(target_vec)
            train_noisy_vectors.append(noisy_vec)
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            nums = list(map(float, parts[1:]))
            target_vec = list(int(c) for c in word)
            noisy_vec = nums
            test_words.append(target_vec)
            test_noisy_vectors.append(noisy_vec)
    padded_train_words, padded_train_vectors, padded_test_words, padded_test_vectors = padding(train_noisy_vectors, train_words, test_noisy_vectors, test_words)
    return normalization(padded_train_words, padded_train_vectors, padded_test_words, padded_test_vectors)

def get_data_loaders(noisy_train_tensor, target_train_tensor, noisy_test_tensor, target_test_tensor):
    n_train = len(noisy_train_tensor)
    train_set = torch.utils.data.TensorDataset(noisy_train_tensor, target_train_tensor)
    N = len(noisy_test_tensor)
    n_test = int(N * 0.7)
    n_val = N - n_test
    test_dataset = torch.utils.data.TensorDataset(noisy_test_tensor, target_test_tensor)
    val_set, test_set = torch.utils.data.random_split(test_dataset, [n_val, n_test])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, pin_memory=True)
    return train_loader, val_loader, test_loader, n_train, n_val, n_test


def main():
    init_seed(42)
    model = DecoderModel.train_model("trainset.txt", "testset.txt")

main()