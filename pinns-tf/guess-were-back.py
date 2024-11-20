import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import h5py


class PINN(Model):
    def __init__(self, layers_dims):
        super(PINN, self).__init__()
        self.network = tf.keras.Sequential()
        for dim in layers_dims:
            self.network.add(layers.Dense(dim, activation="tanh"))
        self.network.add(layers.Dense(1))

    def call(self, inputs):
        return self.network(inputs)


def physics_informed_loss(
    model, x, t, initial_condition, boundary_condition, epsilon
):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        epsilon_tensor = tf.fill(x.shape, tf.constant(epsilon, dtype=tf.float32))
        u = model(tf.concat([x, t, epsilon_tensor], axis=1))

        u_t = tape.gradient(u, t)
        u_x = tape.gradient(u, x)
        u_xx = tape.gradient(u_x, x)

    residual = u_t + u * u_x - epsilon * u_xx

    # Initial condition loss
    initial_loss = tf.reduce_mean(
        tf.square(u[tf.equal(t, tf.reduce_min(t))] - initial_condition)
    )

    # Periodic boundary condition loss
    periodic_loss = tf.reduce_mean(
        tf.square(u[tf.equal(x, tf.reduce_min(x))] - u[tf.equal(x, tf.reduce_max(x))])
    ) + tf.reduce_mean(
        tf.square(
            u_x[tf.equal(x, tf.reduce_min(x))] - u_x[tf.equal(x, tf.reduce_max(x))]
        )
    )

    # Residual loss
    residual_loss = tf.reduce_mean(tf.square(residual))

    return initial_loss + periodic_loss + residual_loss


def train_model(
    model,
    x,
    t,
    initial_condition,
    boundary_condition,
    epsilon,
    epochs,
    learning_rate,
    batch_size,
):
    # Prepare grid for x and t
    x_grid, t_grid = tf.meshgrid(x[:, 0], t[:, 0])  # Create a meshgrid
    x_flat = tf.reshape(x_grid, [-1, 1])  # Flatten to column vector
    t_flat = tf.reshape(t_grid, [-1, 1])  # Flatten to column vector

    # Prepare input for the model
    epsilon_tensor = tf.fill(x_flat.shape, tf.constant(epsilon, dtype=tf.float32))
    input_tensor = tf.concat([x_flat, t_flat, epsilon_tensor], axis=1)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_flat, t_flat, input_tensor))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):
        for x_batch, t_batch, input_batch in dataset:
            with tf.GradientTape() as tape:
                loss = physics_informed_loss(
                    model,
                    x_batch,
                    t_batch,
                    initial_condition,
                    boundary_condition,
                    epsilon,
                )
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")


def predict_solution_iterative(
    model, x, initial_condition, boundary_condition, epsilon, n_steps
):
    """
    Iteratively predict the solution over time.
    Args:
        model: Trained PINN model.
        x: Spatial coordinates as a tensor.
        initial_condition: Initial condition at t=0.
        boundary_condition: Boundary conditions (periodic).
        epsilon: Diffusion parameter.
        n_steps: Number of time steps to predict.
    Returns:
        Array of predictions at each time step.
    """
    dt = 1 / n_steps  # Assume uniform time step
    u = tf.convert_to_tensor(initial_condition, dtype=tf.float32)[:, None]
    all_predictions = [u.numpy().reshape(-1)]

    for i in range(n_steps):
        # Prepare inputs for prediction
        epsilon_tensor = tf.fill(x.shape, tf.constant(epsilon, dtype=tf.float32))[
            :, None
        ]  # Constant epsilon

        u_input = tf.concat([x, u], axis=1)  # Concatenate x and u
        u_input = tf.concat(
            [u_input, epsilon_tensor], axis=1
        )  # Concatenate epsilon_tensor

        u_next = model(u_input)[:, 0]  # Remove extra dimensions

        u_next = apply_periodic_bc(u_next, boundary_condition)
        a = u_next.numpy()
        print(a.shape)
        all_predictions.append(a)
        u = u_next[:, None]
        if u.shape != (1024, 1):
            print(u.shape, i)
    return all_predictions


def apply_periodic_bc(u, boundary_condition):
    """
    Enforce periodic boundary conditions on the solution.
    Args:
        u: Solution array at a given time step (tensor).
        boundary_condition: Precomputed boundary values (not directly used here).
    Returns:
        Solution array with periodic boundary applied (same shape as input).
    """
    # Update the first and last values in `u` to match the periodic boundary
    u = tf.tensor_scatter_nd_update(
        u,
        indices=[[0], [-1]],  # Indices to update: first and last points
        updates=[u[-1], u[0]],  # Periodic values: match last to first, first to last
    )
    return u


# Data loading function
def load_data(file_name):
    with h5py.File(file_name, "r") as f:
        l = list(f.keys())
        initial_condition = []
        boundary_condition = []
        clean_data = []
        epsilon = []
        for i in l[:-1]:
            initial_condition.append(f[f"{i}/initial_condition_noisy"][:])
            boundary_condition.append(f[f"{i}/boundary_condition_noisy"][:])
            clean_data.append(f[f"{i}/clean"][:])
            epsilon.append(f[f"{i}/epsilon"][()])
        initial_condition = np.array(initial_condition)
        boundary_condition = np.array(boundary_condition)
        clean_data = np.array(clean_data)
        epsilon = np.array(epsilon)

        x = f["coords/x-coordinates"][:]
        t = f["coords/t-coordinates"][:-1]
    return (initial_condition, boundary_condition, clean_data, epsilon, x, t)


file_name = "/kaggle/input/burgers-noisy/simulation_data.h5"
initial_condition, boundary_condition, clean_data, epsilon, x, t = load_data(file_name)

x = tf.convert_to_tensor(x, dtype=tf.float32)[:, None]  # x-coordinates as column vector
t = tf.convert_to_tensor(t, dtype=tf.float32)[:, None]  # t-coordinates as column vector
initial_condition = tf.convert_to_tensor(initial_condition, dtype=tf.float32)
boundary_condition = tf.convert_to_tensor(boundary_condition, dtype=tf.float32)

layers_dims = [50, 50, 50]
epochs = 1000
learning_rate = 1e-3
batch_size = 32

model = PINN(layers_dims)

# Train model
train_model(
    model,
    x,
    t,
    initial_condition,
    boundary_condition,
    epsilon,
    epochs,
    learning_rate,
    batch_size,
)

initial_condition, boundary_condition, clean_data, epsilon, x, t = load_data(file_name)
x = tf.convert_to_tensor(x, dtype=tf.float32)[:, None]  # x-coordinates as column vector
t = tf.convert_to_tensor(t, dtype=tf.float32)[:, None]  # t-coordinates as column vector
initial_condition = tf.convert_to_tensor(initial_condition, dtype=tf.float32)
boundary_condition = tf.convert_to_tensor(boundary_condition, dtype=tf.float32)
# Predict solution iteratively
n_steps = 200
predicted_solution = predict_solution_iterative(
    model,
    x=x,
    initial_condition=initial_condition,
    boundary_condition=boundary_condition,
    epsilon=epsilon,
    n_steps=n_steps,
)


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

    #     xcrd = np.load("pde-gen/advection/data/x_coordinate_adv.npy")[:-1]
    #     # print(xcrd.shape)
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
    ani.save("Combo.gif", writer=writer)


visualize_burgers(x.numpy(), np.array(predicted_solution))
