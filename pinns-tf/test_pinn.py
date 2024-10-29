# PINN 2.0 test

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import pickle
from pinn import PhysicsInformedNN

# -----------------------------------------------------------------------------
# Equations to enforce
# -----------------------------------------------------------------------------


@tf.function
def some_eqs(model, coords, eq_params):
    dout = 2
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(coords)
        out = model(coords)
        yp = out[0]
        inv = out[1:]
        fields = [yp[:, jj] for jj in range(dout)]
    df = [tape.gradient(fields[jj], coords) for jj in range(dout)]
    df1dx1 = df[0][:, 0]
    df1dx2 = df[0][:, 1]
    df2dx1 = df[1][:, 0]
    df2dx2 = df[1][:, 1]
    del tape

    eq1 = df1dx1 - np.pi * tf.cos(np.pi * coords[:, 0])
    eq2 = df1dx2 + np.pi * tf.sin(np.pi * coords[:, 1])
    eq3 = df2dx1 - inv[1][:, 0] * coords[:, 0] ** 2
    eq4 = df2dx2 - inv[0][:, 0] * coords[:, 1]

    return [eq1, eq2, eq3, eq4]


# -----------------------------------------------------------------------------
# Generate data
# -----------------------------------------------------------------------------

# Assuming `pde_values` is your data array of shape (101, 1024)
# 101 time steps and 1024 spatial positions
# pde_values = np.random.rand(101, 1024)  # Replace with your actual data

# # Generate spatial positions and time steps
# x_positions = np.linspace(0, 1, 1024)  # Adjust the range as needed
# time_steps = np.linspace(0, 1, 101)  # Adjust the range as needed
path_load = "/home/shusrith/projects/blind-eyes/PredefinedNoisePDE/u,x,t/"
file_to_read = open(path_load + "/2_0.pkl", "rb")
loaded_dictionary = pickle.load(file_to_read)
u = loaded_dictionary["u"]
x = loaded_dictionary["x"]
t = loaded_dictionary["t"]
# Create meshgrid
T, X = np.meshgrid(t, x, indexing="ij")
print(T.shape, X.shape)
# Flatten and combine
X_flat = X.flatten().reshape(-1, 1)
T_flat = T.flatten().reshape(-1, 1)
input_data = np.hstack((X_flat, T_flat)).astype(np.float32)  # Convert to float32
# Flatten PDE values
output_data = (
    u.flatten().reshape(-1, 1).astype(np.float32)
)  # Convert to float32
# Now `input_data` is your X and `output_data` is your Y
X = input_data
Y = output_data
print("Input shape (X):", X.shape)
print("Output shape (Y):", Y.shape)
# -----------------------------------------------------------------------------
# Initialize PINN
# -----------------------------------------------------------------------------
lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, 0.9)
layers = [2] + 2 * [64] + [2]
PINN = PhysicsInformedNN(
    layers,
    dest="./odir/",
    activation="elu",
    optimizer=keras.optimizers.Adam(lr),
    inverse=[{"type": "const", "value": 1.0}, {"type": "const", "value": 1.0}],
    restore=False,
)
PINN.model.summary()

# -----------------------------------------------------------------------------
# Train PINN
# -----------------------------------------------------------------------------
alpha = 0.1
PINN.train(
    X,
    Y,
    some_eqs,
    epochs=2,
    lambda_data=np.array([1.0 for _ in range(len(X))]),
    lambda_phys=np.array([1.0 for _ in range(len(X))]),
    alpha=alpha,
    batch_size=64,
    verbose=True,
    timer=True,
)

t0 = time.time()
tot_eps = 500
PINN.train(
    X,
    Y,
    some_eqs,
    epochs=tot_eps,
    lambda_data=np.array([1.0 for _ in range(len(X))]),
    lambda_phys=np.array([1.0 for _ in range(len(X))]),
    alpha=alpha,
    batch_size=64,
    verbose=False,
    timer=False,
)
print("Time per epoch:", (time.time() - t0) / tot_eps)

# # -----------------------------------------------------------------------------
# # Plot and validate
# # -----------------------------------------------------------------------------

prefix = "odir/fig"
PINN.ckpt.restore(PINN.manager.latest_checkpoint)
if PINN.manager.latest_checkpoint:
    print("Restored from {}".format(PINN.manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

# # -----------------------------------------------------------------------------
# # Make Predictions
# # -----------------------------------------------------------------------------
# # Make predictions
predictions = PINN.model.predict(input_data)

# The first element of the predictions list contains the learned fields
learned_fields = predictions[0]

# If you have inverse parameters, they will be in the subsequent elements of the predictions list
if PINN.inverse is not None:
    inverse_parameters = predictions[1:]
else:
    inverse_parameters = []
PINN.model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), loss="mse")

loss = PINN.model.evaluate(input_data, learned_fields, verbose=1)
# print("Learned fields shape:", learned_fields.shape)
# if inverse_parameters:
#     for i, param in enumerate(inverse_parameters):
#         print(f"Inverse parameter {i} shape:", param.shape)
# Plot loss functions
# ep, lu, lf = np.loadtxt("odir/output.dat", unpack=True)

# ep, c1, c2 = np.loadtxt("odir/inverse.dat", unpack=True)

# plt.figure(0)
# plt.plot(ep, lu, label="Data loss")
# plt.plot(ep, lf, "--", label="Eqs loss")
# plt.legend()
# plt.savefig(prefix + "_0")

# plt.figure(10)
# plt.plot(ep, c1, label="Parameter")
# plt.axhline(2, color="k", ls="--", label="Real value")
# plt.legend()
# plt.savefig(prefix + "_10")

# plt.figure(11)
# plt.plot(ep, c2, label="Parameter")
# plt.axhline(3, color="k", ls="--", label="Real value")
# plt.legend()
# plt.savefig(prefix + "_11")

# Generate points for plotting
x1 = np.linspace(0, 1, 1024).reshape(-1, 1)
x2 = np.linspace(0, 1, 101).reshape(-1, 1)
zs = np.zeros(1024).reshape(-1, 1)

# Evalute along x_2=0
X_plot = np.concatenate((x1, zs), 1)
fields = PINN.model(X_plot)[0]

plt.figure(1)
plt.clf()
plt.plot(x1, fields[:, 0])
plt.plot(x1, np.sin(np.pi * x1) + 1, "ro")
plt.title(r"$f_1(x_1,0)$")
plt.savefig(prefix + "_1")

plt.figure(3)
plt.clf()
plt.plot(x1, fields[:, 1])
plt.plot(x1, x1**3, "ro")
plt.title(r"$f_2(x_1, 0)$")
plt.savefig(prefix + "_3")

# Evalute along x_1=0
X_plot = np.concatenate((zs, x2), 1)
fields = PINN.model(X_plot)[0]

plt.figure(2)
plt.clf()
plt.plot(x2, fields[:, 0])
plt.plot(x2, np.cos(np.pi * x2), "ro")
plt.title(r"$f_1(0, x_2)$")
plt.savefig(prefix + "_2")

plt.figure(4)
plt.clf()
plt.plot(x2, fields[:, 1])
plt.plot(x2, x2**2, "ro")
plt.title(r"$f_2(0, x_2)$")
plt.savefig(prefix + "_4")

# Get gradients

# First convert to tensors
x1 = tf.convert_to_tensor(x1)
x2 = tf.convert_to_tensor(x2)
zs = tf.convert_to_tensor(zs)

# Make tape
with tf.GradientTape(persistent=True) as tape:
    tape.watch([x1, zs])
    X = tf.concat((x1, zs), 1)
    tot = PINN.model(X)[0]
    pred1 = tot[:, 0]
    pred2 = tot[:, 1]
g11 = tape.gradient(pred1, x1)
g21 = tape.gradient(pred2, x1)
del tape

plt.figure(5)
plt.clf()
plt.plot(x1, g11)
plt.plot(x1, np.pi * np.cos(np.pi * x1), "ro")
plt.title(r"$\frac{\partial f_1}{\partial x_1}(x_1,0)$")
plt.savefig(prefix + "_5")

plt.figure(7)
plt.clf()
plt.plot(x1, g21)
plt.plot(x1, 3 * x1**2, "ro")
plt.title(r"$\frac{\partial f_2}{\partial x_1}(x_1,0)$")
plt.savefig(prefix + "_7")

with tf.GradientTape(persistent=True) as tape:
    tape.watch([x2, zs])
    X = tf.concat((zs, x2), 1)
    tot = PINN.model(X)[0]
    pred1 = tot[:, 0]
    pred2 = tot[:, 1]
g12 = tape.gradient(pred1, x2)
g22 = tape.gradient(pred2, x2)
del tape

plt.figure(6)
plt.clf()
plt.plot(x2, g12)
plt.plot(x2, -np.pi * np.sin(np.pi * x2), "ro")
plt.title(r"$\frac{\partial f_1}{\partial x_2}(0, x_2)$")
plt.savefig(prefix + "_6")

plt.figure(8)
plt.clf()
plt.plot(x2, g22)
plt.plot(x2, 2 * x2, "ro")
plt.title(r"$\frac{\partial f_2}{\partial x_2}(0, x_2)$")
plt.savefig(prefix + "_8")

plt.draw()
plt.show()
