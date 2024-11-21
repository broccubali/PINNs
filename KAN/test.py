import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import h5py
import numpy as np

# Load data
with h5py.File("/kaggle/input/burgers-noisy/simulation_data.h5", "r") as f:
    a = list(f.keys())
    clean = []
    noisy = []
    for i in a[:-1]:
        clean.append(f[i]["clean"][:])
        noisy.append(f[i]["noisy"][:])

clean = np.array(clean)
noisy = np.array(noisy)

print("Noisy data shape:", noisy.shape)

# Flatten the input data if necessary
num_samples, height, width = noisy.shape
noisy_flattened = noisy.reshape(num_samples, -1)
clean_flattened = clean.reshape(num_samples, -1)


# Define CombinedLoss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        return self.alpha * self.mse(output, target) + (1 - self.alpha) * self.l1(
            output, target
        )


# Prepare data
X_tensor = torch.tensor(noisy_flattened, dtype=torch.float32)
Y_tensor = torch.tensor(clean_flattened, dtype=torch.float32)

# Create TensorDataset
dataset = TensorDataset(X_tensor, Y_tensor)

# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

# Create DataLoader
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print the shapes to verify
print("Trainloader size:", len(trainloader.dataset))
print("Testloader size:", len(testloader.dataset))


# Define the model with dropout and batch normalization
class KANWithDropout(nn.Module):
    def __init__(self, layers_hidden, dropout_prob=0.3):
        super(KANWithDropout, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.SiLU(),
                    nn.Dropout(dropout_prob),
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Adjust the input dimensions to match the data
input_dim = noisy_flattened.shape[1]  # Flatten the input dimensions
model = KANWithDropout([input_dim, 512, 256, 128, 64, 128, 256, 512, input_dim])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Define ReduceLROnPlateau with patience and min_lr
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.8,
    patience=5,
    min_lr=1e-6,
    verbose=True,
)

# Define loss
criterion = CombinedLoss()

# Training loop
for epoch in range(100):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            val_loss += criterion(output, targets).item()
    val_loss /= len(testloader)

    # Update learning rate
    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}")

# Save the model
torch.save(model.state_dict(), "kan_model.pth")


with h5py.File("/kaggle/input/burgers-noisy/simulation_data.h5", "r") as f:
    a = f["102"]["noisy"][:]
    b = f["102"]["clean"][:]
    x = f["coords"]["x-coordinates"][:]
    f.close()
a = torch.Tensor(a).to(device)
a = a.view(1, -1)
a.shape

# Making predictions
with torch.no_grad():
    u = model(a.to(device)).to("cpu")

# Convert the output to numpy array if needed
u = u.cpu().numpy().reshape((201, 1024))

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tqdm import tqdm


def visualize_burgers(xcrd, data):
    """
    This function animates the Burgers equation

    Args:
    path : path to the desired file
    param: PDE parameter of the data shard to be visualized
    """

    #     xcrd = np.load("burgers/data/x_coordinate.npy")
    # print(xcrd.shape)
    #     data = np.load(path)
    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []

    for i in tqdm(range(data.shape[0])):
        if i == 0:
            im = ax.plot(xcrd, data[i].squeeze(), animated=True, color="blue")
        else:
            im = ax.plot(
                xcrd, data[i].squeeze(), animated=True, color="blue"
            )  # show an initial one first
        ims.append([im[0]])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("burgerCombo.gif", writer=writer)


visualize_burgers(x[:], u)
