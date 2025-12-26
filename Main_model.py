import torch
from torch import nn
device = torch.device('cuda')

epochs = 150
batch = 128
L_mid = 128

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

    def forward(self, x):
        return self.layers(x)

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
    return padded_train_words, padded_train_vectors, padded_test_words, padded_test_vectors, max_vector, max_word

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
    padded_train_words, padded_train_vectors, padded_test_words, padded_test_vectors, L_in, L_out = padding(
        train_noisy_vectors, train_words, test_noisy_vectors, test_words)
    noisy_train_tensor = torch.tensor(padded_train_vectors, dtype=torch.float32)
    target_train_tensor = torch.tensor(padded_train_words, dtype=torch.float32)
    noisy_test_tensor = torch.tensor(padded_test_vectors, dtype=torch.float32)
    target_test_tensor = torch.tensor(padded_test_words, dtype=torch.float32)
    return noisy_train_tensor, target_train_tensor, noisy_test_tensor, target_test_tensor, L_in, L_out

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

def work_with_model(model, train_loader, val_loader, test_loader, n_train, n_val, n_test):
    model = model.to(device)
    criterion = nn.L1Loss()
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
    print("\nL1 на тесте:", test_loss / n_test)

def main():
    noisy_train_tensor, target_train_tensor, noisy_test_tensor, target_test_tensor, L_in, L_out = load_dataset("trainset.txt", "testset.txt")
    model = DecoderModel(L_in, L_out)
    train_loader, val_loader, test_loader, n_train, n_val, n_test = get_data_loaders(noisy_train_tensor, target_train_tensor, noisy_test_tensor, target_test_tensor)
    work_with_model(model, train_loader, val_loader, test_loader, n_train, n_val, n_test)

main()