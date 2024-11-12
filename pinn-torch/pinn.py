import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        u = self.hidden(inputs)
        return u


def residual_loss(model, x, t, nu):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0]
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0]

    f = u_t + u * u_x - nu * u_xx
    return torch.mean(f**2)


def energy_loss(model, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0]

    energy = torch.mean(u**2 + u_x**2 + u_xx**2)
    return energy


def data_loss(model, x_data, t_data, u_data):
    u_pred = model(x_data, t_data)
    return torch.mean((u_pred - u_data) ** 2)


def train_pinn(model, optimizer, x_res, t_res, x_data, t_data, u_data, nu, epochs):
    model.to(device)
    x_res, t_res = x_res.to(device), t_res.to(device)
    x_data, t_data, u_data = x_data.to(device), t_data.to(device), u_data.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        res_loss = residual_loss(model, x_res, t_res, nu)
        en_loss = energy_loss(model, x_res, t_res)
        d_loss = data_loss(model, x_data, t_data, u_data)
        loss = res_loss + en_loss + d_loss
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}, Total Loss: {loss.item()}, Residual Loss: {res_loss.item()}, Energy Loss: {en_loss.item()}, Data Loss: {d_loss.item()}"
            )

    print(
        f"Final Loss: {loss.item()}, Residual Loss: {res_loss.item()}, Energy Loss: {en_loss.item()}, Data Loss: {d_loss.item()}"
    )
    return loss.item()


def predict(model, x, t):
    model.eval()
    with torch.no_grad():
        u_pred = model(x, t)
    return u_pred


def mean_absolute_error(predicted, true):
    return torch.mean(torch.abs(predicted - true))


def normalized_mse(predicted, true):
    return torch.mean((predicted - true) ** 2) / torch.mean(
        (true - torch.mean(true)) ** 2
    )


# Define the viscosity coefficient for Burgers' equation
nu = 0.01

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# path_load = "/home/shusrith/projects/blind-eyes/PredefinedNoisePDE/u,x,t/"
# with open(path_load + "/3_0.pkl", "rb") as file_to_read:
#     loaded_dictionary = pickle.load(file_to_read)
u = np.load("/home/shusrith/projects/blind-eyes/PINNs/pde-gen/data/burgerNoisy.npy")

# u = loaded_dictionary["u"]
# x = loaded_dictionary
# t = loaded_dictionary["t"]
x = np.load("/home/shusrith/projects/blind-eyes/PINNs/pde-gen/data/x_coordinate.npy")
t = np.load("/home/shusrith/projects/blind-eyes/PINNs/pde-gen/data/t_coordinate.npy")[:-1]
X, T = np.meshgrid(x, t)

x_data = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1)
t_data = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1)
u_data = torch.tensor(u.flatten(), dtype=torch.float32).view(-1, 1)

N_res = 1000
x_res = torch.rand(N_res, 1) * 2 - 1
t_res = torch.rand(N_res, 1) * 2 - 1

final_loss = train_pinn(
    model, optimizer, x_res, t_res, x_data, t_data, u_data, nu, epochs=1000
)

# with open(path_load + "/3_1.pkl", "rb") as file_to_read:
#     loaded_dictionary = pickle.load(file_to_read)

# u_test = loaded_dictionary["u"]
# x_test = loaded_dictionary["x"]
# t_test = loaded_dictionary["t"]

u_test = np.load("/home/shusrith/projects/blind-eyes/PINNs/pde-gen/data/burgerClean.npy")
x_test = np.load("/home/shusrith/projects/blind-eyes/PINNs/pde-gen/data/x_coordinate.npy")
t_test = np.load("/home/shusrith/projects/blind-eyes/PINNs/pde-gen/data/t_coordinate.npy")

X_test, T_test = np.meshgrid(x_test, t_test)

x_data_test = torch.tensor(X_test.flatten(), dtype=torch.float32).view(-1, 1)
t_data_test = torch.tensor(T_test.flatten(), dtype=torch.float32).view(-1, 1)
u_data_test = torch.tensor(u_test.flatten(), dtype=torch.float32).view(-1, 1)

x_data_test, t_data_test, u_data_test = (
    x_data_test.to(device),
    t_data_test.to(device),
    u_data_test.to(device),
)

u_pred = predict(model, x_data_test, t_data_test)

u_pred = u_pred.cpu().detach()

plt.figure(figsize=(10, 6))
plt.imshow(
    u_pred.reshape(X_test.shape),
    extent=[-1, 1, 0, 1],
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
plt.colorbar(label="Predicted u(x, t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Predicted Solution of Burgers' Equation with PINN")
plt.show()

mae_loss = mean_absolute_error(u_pred, u_data_test.cpu())
print(f"Prediction MAE Loss: {mae_loss.item()}")

nmse_loss = normalized_mse(u_pred, u_data_test.cpu())
print(f"Prediction NMSE Loss: {nmse_loss.item()}")
