import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import diags
import scipy.linalg
import tensorflow as tf

# Function to generate grid points
def generate_grid(grid_size=(32, 32)):
    x = np.linspace(0, 1, grid_size[0])
    y = np.linspace(0, 1, grid_size[1])
    X, Y = np.meshgrid(x, y)
    return X, Y

# Function to add noise to data
def add_noise(data, noise_level=0.1):
    additive_noise = noise_level * np.random.randn(*data.shape)
    multiplicative_noise = noise_level * data * np.random.randn(*data.shape)
    random_perturbation = noise_level * (np.random.rand(*data.shape) - 0.5)
    noisy_data = data + additive_noise + multiplicative_noise + random_perturbation
    return noisy_data

# Generate elliptic PDE data with noise (Poisson's Equation)
def generate_elliptic_data(grid_size=(32, 32), num_samples=100, noise_level=0.1):
    data_info = []
    data_samples = []

    for i in range(num_samples):
        X, Y = generate_grid(grid_size)
        f = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)  # Source term
        
        N = grid_size[0]
        diagonals = [-4 * np.ones(N), np.ones(N - 1), np.ones(N - 1)]
        laplacian = diags(diagonals, [0, -1, 1], shape=(N, N)).toarray()
        laplacian_2d = np.kron(np.eye(N), laplacian) + np.kron(laplacian, np.eye(N))
        
        f_flat = f.flatten()
        u_flat = scipy.linalg.solve(laplacian_2d, f_flat)
        u = u_flat.reshape(grid_size)
        
        # Add noise to the solution
        noisy_u = add_noise(u, noise_level=noise_level)

        data_samples.append((X, Y, noisy_u))
        data_info.append({
            'Sample ID': i,
            'PDE Type': 'Elliptic',
            'Original Mean': np.mean(u),
            'Noisy Mean': np.mean(noisy_u),
            'Original Std': np.std(u),
            'Noisy Std': np.std(noisy_u)
        })

    return data_samples, pd.DataFrame(data_info)

# Generate parabolic PDE data with noise (Heat Equation)
def generate_parabolic_data(grid_size=(32, 32), num_samples=100, time_steps=50, noise_level=0.1):
    data_info = []
    data_samples = []

    for i in range(num_samples):
        X, Y = generate_grid(grid_size)
        u = np.random.rand(*grid_size)  # Initial condition
        
        dt = 0.01
        dx = 1 / grid_size[0]
        alpha = dt / (dx**2)
        
        for _ in range(time_steps):
            u_new = u.copy()
            u_new[1:-1, 1:-1] += alpha * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1])
            u = u_new
        
        # Add noise to the solution
        noisy_u = add_noise(u, noise_level=noise_level)

        data_samples.append((X, Y, noisy_u))
        data_info.append({
            'Sample ID': i,
            'PDE Type': 'Parabolic',
            'Original Mean': np.mean(u),
            'Noisy Mean': np.mean(noisy_u),
            'Original Std': np.std(u),
            'Noisy Std': np.std(noisy_u)
        })

    return data_samples, pd.DataFrame(data_info)

# Generate hyperbolic PDE data with noise (Wave Equation)
def generate_hyperbolic_data(grid_size=(32, 32), num_samples=100, time_steps=50, noise_level=0.1):
    data_info = []
    data_samples = []

    for i in range(num_samples):
        X, Y = generate_grid(grid_size)
        u = np.zeros(grid_size)
        u_prev = np.zeros(grid_size)
        u_prev[grid_size[0] // 2, grid_size[1] // 2] = 1  # Initial disturbance at center
        
        dt = 0.01
        dx = 1 / grid_size[0]
        c = 1  # Wave speed
        alpha = (c * dt / dx)**2
        
        for _ in range(time_steps):
            u_new = np.zeros(grid_size)
            u_new[1:-1, 1:-1] += (
                2 * u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
                alpha * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1])
            )
            u_prev[:] = u[:]
            u[:] = u_new[:]
        
        # Add noise to the solution
        noisy_u = add_noise(u_new.copy(), noise_level=noise_level)

        data_samples.append((X,Y,noisy_u))
        data_info.append({
            'Sample ID': i,
            'PDE Type': 'Hyperbolic',
            'Original Mean': np.mean(u),
            'Noisy Mean': np.mean(noisy_u),
            'Original Std': np.std(u),
            'Noisy Std': np.std(noisy_u)
        })

    return data_samples,pd.DataFrame(data_info)

# Define PINN model for Elliptic PDEs
class PINN_Elliptic(tf.keras.Model):
    def __init__(self):
        super(PINN_Elliptic,self).__init__()
        self.hidden_layers=[tf.keras.layers.Dense(100 ,activation='tanh') for _ in range(8)]
        self.output_layer=tf.keras.layers.Dense(1)

    def call(self , inputs):
      x=inputs
      for layer in self.hidden_layers:
          x=layer(x)
      return self.output_layer(x)

# Loss function for Elliptic PDEs
def elliptic_pde_loss(model , x , y):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x,y])
        
        # Prediction from model 
        u_pred=model(tf.concat([x,y], axis=1))
        
         # Compute gradients
         # First derivatives with respect to x and y 
        u_x=tape.gradient(u_pred,x)
        u_y=tape.gradient(u_pred,y)

         # Second derivatives 
        u_xx=tape.gradient(u_x,x)
        u_yy=tape.gradient(u_y,y)

         # Poisson's equation residual: Δu - f(x,y) 
        pde_residual=u_xx+u_yy  
         
        return tf.reduce_mean(tf.square(pde_residual))

# Define PINN model for Parabolic PDEs
class PINN_Parabolic(tf.keras.Model):
    def __init__(self):
      super(PINN_Parabolic,self).__init__()
      self.hidden_layers=[tf.keras.layers.Dense(100 ,activation='tanh') for _ in range(8)]
      self.output_layer=tf.keras.layers.Dense(1)

    def call(self , inputs):
      x=inputs
      for layer in self.hidden_layers:
          x=layer(x)
      return self.output_layer(x)

# Loss function for Parabolic PDEs
def parabolic_pde_loss(model , x , y , t):
     with tf.GradientTape(persistent=True) as tape:
         tape.watch([x,y,t])
         
         # Prediction from model 
         u_pred=model(tf.concat([x,y,t], axis=1))

         # First derivative with respect to time t 
         u_t=tape.gradient(u_pred,t)

         # Second derivatives with respect to space x and y 
         u_xx=tape.gradient(tape.gradient(u_pred,x),x)
         u_yy=tape.gradient(tape.gradient(u_pred,y),y)

         # Heat equation residual: ∂u/∂t - Δu 
         pde_residual=u_t-(u_xx+u_yy)

         return tf.reduce_mean(tf.square(pde_residual))

# Define PINN model for Hyperbolic PDEs
class PINN_Hyperbolic(tf.keras.Model):
    def __init__(self):
      super(PINN_Hyperbolic,self).__init__()
      self.hidden_layers=[tf.keras.layers.Dense(100 ,activation='tanh') for _ in range(8)]
      self.output_layer=tf.keras.layers.Dense(1)

    def call(self , inputs):
      x=inputs
      for layer in self.hidden_layers:
          x=layer(x)
      return self.output_layer(x)

# Loss function for Hyperbolic PDEs
def hyperbolic_pde_loss(model , x , y , t):
     with tf.GradientTape(persistent=True) as tape:
         tape.watch([x,y,t])
         
         # Prediction from model 
         u_pred=model(tf.concat([x,y,t], axis=1))

         # Second derivative with respect to time t 
         u_tt=tape.gradient(tape.gradient(u_pred,t),t)

         # Second derivatives with respect to space x and y 
         u_xx=tape.gradient(tape.gradient(u_pred,x),x)
         u_yy=tape.gradient(tape.gradient(u_pred,y),y)

         # Wave equation residual: ∂²u/∂t² - Δu 
         pde_residual=u_tt-(u_xx+u_yy)

         return tf.reduce_mean(tf.square(pde_residual))

# Generate datasets for all three types of PDEs 
elliptic_data , elliptic_summary   = generate_elliptic_data(noise_level=0.3,num_samples=5)
parabolic_data , parabolic_summary   = generate_parabolic_data(noise_level=0.3,num_samples=5,time_steps=50)
hyperbolic_data , hyperbolic_summary   = generate_hyperbolic_data(noise_level=0.3,num_samples=5,time_steps=50)

# Display a sample of the noisy data for visualization
plt.figure(figsize=(15 ,5))

# Elliptic PDE visualization
plt.subplot(131)
plt.contourf(elliptic_data[0][0], elliptic_data[0][1], elliptic_data[0][2], cmap='viridis')
plt.title('Noisy Elliptic PDE Solution')
plt.colorbar()

# Parabolic PDE visualization
plt.subplot(132)
plt.contourf(parabolic_data[0][0], parabolic_data[0][1], parabolic_data[0][2], cmap='viridis')
plt.title('Noisy Parabolic PDE Solution')
plt.colorbar()

# Hyperbolic PDE visualization
plt.subplot(133)
plt.contourf(hyperbolic_data[0][0], hyperbolic_data[0][1], hyperbolic_data[0][2], cmap='viridis')
plt.title('Noisy Hyperbolic PDE Solution')
plt.colorbar()

plt.tight_layout()
plt.show()

# Display tabular summary of the results 
print("Elliptic PDE Data Summary:")
print(elliptic_summary) 
print("\nParabolic PDE Data Summary:")
print(parabolic_summary) 
print("\nHyperbolic PDE Data Summary:")
print(hyperbolic_summary)


epochs_range=range(100) 

elliptic_loss_values=[]
parabolic_loss_values=[]
hyperbolic_loss_values=[]

# Training Loop Simulation (for demonstration purposes only; no actual optimization is performed here).
for epoch in epochs_range:
    
     # Simulate input values for elliptic training (random values within the domain).
     x_train_elliptic=np.random.uniform(0 , 1 , (100 , 1)).astype(np.float32)  
     y_train_elliptic=np.random.uniform(0 , 1 , (100 , 1)).astype(np.float32)  
     
     # Create an instance of the elliptic PINN model.
     elliptic_model=PINN_Elliptic() 

     # Convert NumPy arrays to TensorFlow tensors
     x_tensor_elliptic=tf.convert_to_tensor(x_train_elliptic)
     y_tensor_elliptic=tf.convert_to_tensor(y_train_elliptic)

     # Calculate loss using the elliptic loss function.
     loss_value=elliptic_pde_loss(elliptic_model,x_tensor_elliptic,y_tensor_elliptic) 
     elliptic_loss_values.append(loss_value.numpy())

     # Simulate input values for parabolic training.
     x_train_parabolic=np.random.uniform(0 , 10 , (100 , 3)).astype(np.float32)  
     y_train_parabolic=np.random.uniform(0 ,10,(100 ,3)).astype(np.float32)  
     t_train_parabolic=np.random.uniform(0 ,10,(100 ,3)).astype(np.float32)  

     parabolic_model=PINN_Parabolic() 

     # Convert NumPy arrays to TensorFlow tensors
     x_tensor_parabolic=tf.convert_to_tensor(x_train_parabolic)
     y_tensor_parabolic=tf.convert_to_tensor(y_train_parabolic)
     t_tensor_parabolic=tf.convert_to_tensor(t_train_parabolic)

     loss_value_parabolic=parabolic_pde_loss(parabolic_model,x_tensor_parabolic,y_tensor_parabolic,t_tensor_parabolic) 
     parabolic_loss_values.append(loss_value_parabolic.numpy())

     # Simulate input values for hyperbolic training.
     x_train_hyperbolic=np.random.uniform(0 ,10,(100 ,3)).astype(np.float32)  
     y_train_hyperbolic=np.random.uniform(0 ,10,(100 ,3)).astype(np.float32)  
     t_train_hyperbolic=np.random.uniform(0 ,10,(100 ,3)).astype(np.float32)  

     hyperbolic_model=PINN_Hyperbolic() 

     # Convert NumPy arrays to TensorFlow tensors
     x_tensor_hyperbolic=tf.convert_to_tensor(x_train_hyperbolic)
     y_tensor_hyperbolic=tf.convert_to_tensor(y_train_hyperbolic)
     t_tensor_hyperbolic=tf.convert_to_tensor(t_train_hyperbolic)

     loss_value_hyperbolic=hyperbolic_pde_loss(hyperbolic_model,x_tensor_hyperbolic,y_tensor_hyperbolic,t_tensor_hyperbolic) 
     hyperbolic_loss_values.append(loss_value_hyperbolic.numpy())

# Combined Training Loss Visualization 
plt.figure(figsize=(15 ,5))
plt.plot(epochs_range , elliptic_loss_values,label='Elliptical Loss',color='blue')
plt.plot(epochs_range , parabolic_loss_values,label='Parabola Loss',color='orange')
plt.plot(epochs_range , hyperbolic_loss_values,label='Hyperbola Loss',color='green')
plt.title('Combined Training Loss for All PDEs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()