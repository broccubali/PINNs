import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import h5py
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm


@tf.keras.utils.register_keras_serializable()
class PINN(Model):
    def __init__(self, layers_dims):
        super(PINN, self).__init__()
        self.network = tf.keras.Sequential()
        for dim in layers_dims:
            self.network.add(layers.Dense(dim, activation="tanh"))
        self.network.add(layers.Dense(1))  # Single scalar output per sample

    def call(self, inputs):
        return self.network(inputs)


# Dataset creation
def create_sampled_dataset(
    initial_conditions, boundary_conditions, epsilons, batch_size
):
    dataset = tf.data.Dataset.from_tensor_slices(
        (initial_conditions, boundary_conditions, epsilons)
    )
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset


# Loss function
def physics_informed_loss(model, initial_condition, boundary_condition, epsilon):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(initial_condition)
        epsilon_tensor = tf.expand_dims(epsilon, axis=-1)  # Match shape
        print(initial_condition.shape, boundary_condition.shape, epsilon_tensor.shape)
        a = tf.concat([initial_condition, boundary_condition, epsilon_tensor], axis=1)
        u = model(a)
        print(u.shape)
        # Compute gradients
        u_x = tape.gradient(u, initial_condition)
        u_xx = tape.gradient(u_x, initial_condition)

    # Residual loss
    residual = u * u_x - epsilon_tensor * u_xx
    residual_loss = tf.reduce_mean(tf.square(residual))

    # Initial condition loss
    initial_loss = tf.reduce_mean(tf.square(u - initial_condition))

    # Periodic boundary condition loss
    periodic_loss = tf.reduce_mean(tf.square(u[:, 0] - u[:, -1])) + tf.reduce_mean(
        tf.square(u_x[:, 0] - u_x[:, -1])
    )

    return initial_loss + periodic_loss + residual_loss


# Training function
def train_model(model, dataset, epochs, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):
        for initial_batch, boundary_batch, epsilon_batch in dataset:
            with tf.GradientTape() as tape:
                loss = physics_informed_loss(
                    model, initial_batch, boundary_batch, epsilon_batch
                )
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")


import tensorflow as tf
import numpy as np


def predict_solution_iterative_with_bc(
    model, initial_condition, epsilon, n_steps, boundary_condition
):
    """
    Iteratively predict the solution over time, applying periodic boundary conditions.
    Args:
        model: Trained PINN model.
        initial_condition: Initial condition at t=0 (1D tensor with shape [1024, 1]).
        epsilon: Diffusion parameter (scalar).
        n_steps: Number of time steps to predict.
        boundary_condition: Boundary condition tensor (1D tensor with shape [1028]).
    Returns:
        Array of predictions at each time step.
    """
    # Reshape initial_condition and boundary_condition to match input shape [1, 2053]
    u = tf.convert_to_tensor(initial_condition, dtype=tf.float32)[
        :, None
    ]  # [1024, 1] -> [1, 1024]

    # Ensure boundary_condition has the correct shape [1, 1028]
    boundary_condition = tf.convert_to_tensor(
        boundary_condition, dtype=tf.float32
    )  # [1028]
    boundary_condition = tf.reshape(boundary_condition, [1, -1])  # [1028] -> [1, 1028]

    # Expand epsilon to match shape [1, 1]
    epsilon_tensor = tf.fill([1, 1], tf.constant(epsilon, dtype=tf.float32))  # [1, 1]

    # Concatenate the tensors along axis 1 to form a shape of [1, 2053]
    u_input = tf.concat(
        [u, boundary_condition, epsilon_tensor], axis=1
    )  # [1, 1024] + [1, 1028] + [1, 1] -> [1, 2053]

    # Initialize list to store predictions
    all_predictions = []

    for _ in range(n_steps):
        # Predict next time step
        u_next = model(
            u_input
        )  # Assuming model output is [1, 1] (we remove extra dimension)

        # Apply periodic boundary conditions
        u_next = apply_periodic_bc(u_next)  # Ensure periodic BC

        # Store the result and update u_input (we reshape for the next iteration)
        all_predictions.append(u_next.numpy().reshape(-1))

        # Update u_input for the next iteration
        u_input = tf.concat(
            [u_next[:, None], boundary_condition, epsilon_tensor], axis=1
        )  # [1, 2053]

    return np.array(all_predictions)


def apply_periodic_bc(u):
    """
    Enforce periodic boundary conditions on the solution.
    Args:
        u: Solution array at a given time step (tensor).
    Returns:
        Solution array with periodic boundary applied (same shape as input).
    """
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

# Convert to tensors
initial_condition = tf.convert_to_tensor(initial_condition, dtype=tf.float32)
boundary_condition = tf.convert_to_tensor(boundary_condition, dtype=tf.float32)
epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)

layers_dims = [50, 50, 50]
epochs = 1000
learning_rate = 1e-3
batch_size = 32
model = PINN(layers_dims)

# Create dataset
dataset = create_sampled_dataset(
    initial_condition, boundary_condition, epsilon, batch_size
)

# Train the model
train_model(model, dataset, epochs, learning_rate)


# Save the trained PINN model

model.save("pinn_model.keras")
# model = tf.keras.models.load_model("/kaggle/input/pinn/tensorflow2/default/1/pinn_model.keras")


import tensorflow as tf
import numpy as np


def predict_solution_iterative_with_bc(
    model, initial_condition, epsilon, n_steps, boundary_condition
):
    """
    Iteratively predict the solution over time, applying periodic boundary conditions.
    Args:
        model: Trained PINN model.
        initial_condition: Initial condition at t=0 (1D tensor with shape [1024, 1]).
        epsilon: Diffusion parameter (scalar).
        n_steps: Number of time steps to predict.
        boundary_condition: Boundary condition tensor (1D tensor with shape [1028]).
    Returns:
        Array of predictions at each time step.
    """
    u = tf.convert_to_tensor(initial_condition, dtype=tf.float32)[
        :, None
    ]  # [1024, 1] -> [1, 1024]
    u = tf.reshape(u, [1, -1])
    boundary_condition = tf.convert_to_tensor(
        boundary_condition, dtype=tf.float32
    )  # [1028]
    boundary_condition = tf.reshape(boundary_condition, [1, -1])  # [1028] -> [1, 1028]
    epsilon_tensor = tf.fill([1, 1], tf.constant(epsilon, dtype=tf.float32))  # [1, 1]
    u_input = tf.concat(
        [u, boundary_condition, epsilon_tensor], axis=1
    )  # [1, 1024] + [1, 1028] + [1, 1] -> [1, 2053]

    # Initialize list to store predictions
    all_predictions = []

    for _ in range(n_steps):
        u_next = model(
            u_input
        )  # Assuming model output is [1, 1] (we remove extra dimension)
        print(u_next.shape)
        u_next = apply_periodic_bc(u_next)
        all_predictions.append(u_next.numpy().reshape(-1))
        u_input = tf.concat(
            [u_next[:], boundary_condition, epsilon_tensor], axis=1
        )  # [1, 2053]

    return np.array(all_predictions)


def apply_periodic_bc(u):
    """
    Enforce periodic boundary conditions on the solution.
    Args:
        u: Solution array at a given time step (tensor).
    Returns:
        Solution array with periodic boundary applied (same shape as input).
    """
    u = tf.tensor_scatter_nd_update(
        u,
        indices=[[0], [-1]],  # Indices to update: first and last points
        updates=[u[-1], u[0]],  # Periodic values: match last to first, first to last
    )
    return u


# Predict solution iteratively
n_steps = 200
predicted_solution = predict_solution_iterative_with_bc(
    model, initial_condition[0], epsilon[0], n_steps, boundary_condition[0]
)

visualize_burgers(x, predicted_solution)
