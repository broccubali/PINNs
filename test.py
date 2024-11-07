from kan import KAN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load the training data from the .pkl file
path_load = "/home/shusrith/projects/blind-eyes/PredefinedNoisePDE/u,x,t/"
file_to_read = open(path_load + "3_0.pkl", "rb")
loaded_dictionary = pickle.load(file_to_read)
file_to_read.close()

u_noisy = loaded_dictionary["u_noisy"]
u = loaded_dictionary["u"]


# Flatten the data
input_data = u_noisy.flatten().reshape(-1, 1).astype(np.float32)  # Convert to float32
output_data = u.flatten().reshape(-1, 1).astype(np.float32)  # Convert to float32

# Convert to PyTorch tensors
X_tensor = torch.tensor(input_data)
Y_tensor = torch.tensor(output_data)

# Create TensorDataset
dataset = TensorDataset(X_tensor, Y_tensor)

# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

# Create DataLoader
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Print the shapes to verify
print("Trainloader size:", len(trainloader.dataset))
print("Testloader size:", len(testloader.dataset))

# Define model
model = KAN([3, 32, 64, 1])  # Adjust input and output sizes as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)
# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define loss
criterion = nn.MSELoss()  # Use MSELoss for regression tasks

# Training loop
for epoch in range(10):
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
    scheduler.step()

    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}")

# Load the new data from the .pkl file for prediction
file_to_read = open(path_load + "3_1.pkl", "rb")
loaded_dictionary = pickle.load(file_to_read)
file_to_read.close()

u_noisy_new = loaded_dictionary["u_noisy"]
u_new = loaded_dictionary["u"]
x = loaded_dictionary["x"]
t_new = loaded_dictionary["t"]

# Flatten the new data
input_data_new = (
    u_noisy_new.flatten().reshape(-1, 1).astype(np.float32)
)  # Convert to float32
output_data_new = (
    u_new.flatten().reshape(-1, 1).astype(np.float32)
)  # Convert to float32

# Convert to PyTorch tensors
X_tensor_new = torch.tensor(input_data_new).to(device)
Y_tensor_new = torch.tensor(output_data_new).to(device)

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(X_tensor_new)

# Calculate loss on predictions
prediction_loss = criterion(predictions, Y_tensor_new).item()

# Print the prediction loss
print(f"Prediction Loss: {prediction_loss}")

# Convert predictions to numpy array and reshape to original shape
predictions = predictions.cpu().numpy().reshape(u_noisy_new.shape)

# Print or save the predictions as needed
fig, ax = plt.subplots()
(line,) = ax.plot(x, predictions[0])


def update(frame):
    line.set_ydata(predictions[frame])
    ax.set_title(f"Time step {frame}")
    return (line,)


# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=predictions.shape[0], blit=True
)

# Save as GIF
gif_path = "pls.gif"
ani.save(gif_path, writer="imagemagick", fps=10)
