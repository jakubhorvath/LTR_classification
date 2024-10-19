import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append("../../")
from utils.CNN_utils import remove_N, onehote

from torch import nn
# Define the PyTorch model
class Conv1DModel(nn.Module):
    def __init__(self):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=16)  # same padding
        self.dropout1 = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4)  # same padding
        self.dropout2 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (4000 // 16), 256)  # Adjust input size accordingly
        self.fc2 = nn.Linear(256, 13)
        self.softmax = nn.Softmax(1)

    def preprocess(self, x):
        """
            One-hot encoding and removing Ns from the sequences
        """
        x = [remove_N(seq) for seq in x]
        x = [onehote(seq) for seq in x]
        return x

    def forward(self, x):
        x = x.to("cuda")

        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class CNN_dataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def train_model(model, X, y, criterion, optimizer, num_epochs=15, patience=3):
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X.transpose(1, 2))  # PyTorch expects channels first, so we transpose
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

def eval_model(model, X, y, criterion):
    val_dataset = TensorDataset(X, y)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_X, val_Y in val_loader:
            val_outputs = model(val_X.transpose(1, 2))
            val_loss += criterion(val_outputs, val_Y).item()

    val_loss /= len(val_loader)
    print(f'Validation Loss: {val_loss}')

    return val_loss