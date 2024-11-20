import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import h5py


class PINN(Model):
    def __init__(self, layers_dims, epsilon):
        super(PINN, self).__init__()
        self.epsilon = epsilon

        self.network = tf.keras.Sequential()
        for dim in layers_dims:
            self.network.add(layers.Dense(dim, activation="tanh"))
        self.network.add(layers.Dense(1))

    def call(self, inputs):
        return self.network(inputs)


def physics_informed_loss(
    model, x, t, initial_condition, boundary_condition, du, epsilon
):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u = model(tf.concat([x, t], axis=1))

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
    du,
    epsilon,
    epochs,
    learning_rate,
):
    # Prepare grid for x and t
    x_grid, t_grid = tf.meshgrid(x[:, 0], t[:, 0])  # Create a meshgrid
    x_flat = tf.reshape(x_grid, [-1, 1])  # Flatten to column vector
    t_flat = tf.reshape(t_grid, [-1, 1])  # Flatten to column vector

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = physics_informed_loss(
                model,
                x_flat,
                t_flat,
                initial_condition,
                boundary_condition,
                du,
                epsilon,
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")


# Predict solution
def predict_solution(model, x, t):
    xt = tf.concat([x, t], axis=1)
    return model(xt)


# Data loading function
def load_data(file_name):
    with h5py.File(file_name, "r") as f:
        initial_condition = f["0/initial_condition_noisy"][:]
        boundary_condition = f["0/boundary_condition_noisy"][:]
        clean_data = f["0/clean"][:]
        du = f["0/du"][()]
        epsilon = f["0/epsilon"][()]
        u0 = f["0/u0"][()]
        x = f["coords/x-coordinates"][:]
        t = f["coords/t-coordinates"][:-1]
    return (
        initial_condition,
        boundary_condition,
        clean_data,
        du,
        epsilon,
        u0,
        x,
        t,
    )


# Load the data
file_name = (
    "/kaggle/input/burgers-noisy/simulation_data.h5"  # Replace with your data file
)
initial_condition, boundary_condition, clean_data, du, epsilon, u0, x, t = load_data(
    file_name
)

# Convert data to tensors
x = tf.convert_to_tensor(x, dtype=tf.float32)[:, None]  # x-coordinates as column vector
t = tf.convert_to_tensor(t, dtype=tf.float32)[:, None]  # t-coordinates as column vector
initial_condition = tf.convert_to_tensor(initial_condition, dtype=tf.float32)
boundary_condition = tf.convert_to_tensor(boundary_condition, dtype=tf.float32)

# Define model parameters
layers_dims = [50, 50, 50]
epochs = 10000
learning_rate = 1e-3

# Initialize model
model = PINN(layers_dims, epsilon)

# Train model
train_model(
    model,
    x,
    t,
    initial_condition,
    boundary_condition,
    du,
    epsilon,
    epochs,
    learning_rate,
)


# Predict the solution for the time range
x_pred = tf.convert_to_tensor(
    np.linspace(x.numpy().min(), x.numpy().max(), len(x))[:, None], dtype=tf.float32
)
t_pred = tf.convert_to_tensor(
    np.linspace(t.numpy().min(), t.numpy().max(), len(t))[:, None], dtype=tf.float32
)
u_pred = predict_solution(model, x_pred, t_pred)

# u_pred will contain the solution over the predicted time steps
print("Solution predicted.")
