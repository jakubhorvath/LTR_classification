import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append("../../")
from utils.CNN_utils import remove_N, onehote

# Define the PyTorch model
class Conv1DModel(nn.Module):
    def __init__(self, inp_shape):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inp_shape, out_channels=32, kernel_size=16, padding=8)  # same padding
        self.dropout1 = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, padding=2)  # same padding
        self.dropout2 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (inp_shape // 16), 256)  # Adjust input size accordingly
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def preprocess(self, x):
        """
            One-hot encoding and removing Ns from the sequences
        """
        x = [remove_N(seq) for seq in x]
        x = [onehote(seq) for seq in x]
        return x

    def forward(self, x):
        #x = self.preprocess(x)

        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


sys.exit()

# Instantiate the model
model = Conv1DModel()

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters())

# Convert numpy arrays to PyTorch tensors
trainX_tensor = torch.tensor(trainX, dtype=torch.float32)
trainY_tensor = torch.tensor(trainY, dtype=torch.float32).reshape(-1, 1)
valX_tensor = torch.tensor(valX, dtype=torch.float32)
valY_tensor = torch.tensor(valY, dtype=torch.float32).reshape(-1, 1)

# Create DataLoader instances
train_dataset = TensorDataset(trainX_tensor, trainY_tensor)
val_dataset = TensorDataset(valX_tensor, valY_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training loop
num_epochs = 15
patience = 3
early_stopping_counter = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X.transpose(1, 2))  # PyTorch expects channels first, so we transpose
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_X, val_Y in val_loader:
            val_outputs = model(val_X.transpose(1, 2))
            val_loss += criterion(val_outputs, val_Y).item()

    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        # Save the model checkpoint
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print('Early stopping')
            break

# To load the model later:
# model.load_state_dict(torch.load('best_model.pth'))
# model.eval()
