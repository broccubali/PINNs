# Dataset Generation

The code has been adopted from https://github.com/pdebench/PDEBench and necessary modifications have been made to add noise to the data. Burger's equation has been used as an example to show proof of concept. Noise added is a random choice between skewed normal and exponential noise. 

## Initial Conditions 

This function initializes the initial condition of the PDE based on the specified mode.


- if mode is `sin`, the initial condition is a sine wave.
- if mode is `gaussian`, the initial condition is a Gaussian distribution.
- if mode is `step`, the initial condition is a step function.
- if mode is `possin`, the initial condition is a positive sine wave, in which the entire sine wave stays above zero
- if mode is `sinsin`, the initial condition is a double sine wave, with each wave having a different frequency


## Courant Functions 

The Courant number, also known as the Courant-Friedrichs-Lewy (CFL) number, is a dimensionless number that plays a crucial role in the stability of numerical solutions to partial differential equations (PDEs). The Courant number is a critical parameter for ensuring the stability of numerical solutions to PDEs. It is used to determine appropriate time step sizes relative to the spatial grid size and the characteristic velocity or diffusion coefficient of the problem.

`Courant` function calculates the Courant number for the given velocity field u and grid spacing dx.

`Courant_diff` function calculates the Courant number for diffusion with the given grid spacing dx and diffusion coefficient epsilon.

## Noise Generation

This function generates a noise array with random variability, to give us noisy data of different forms, to increase the robustness of the model to the noise. 

  - Randomly chooses between two noise types: **skew normal** or **exponential** distribution.  
  - If a random value `a` < 0.5, uses  a skew normal distribution with skewness randomly selected between -5 and 5.  
  - Otherwise, uses an exponential distribution with a scale randomly selected between 0 and 4.  
- It returns the generated noise scaled by `noise_level`.  
- It adds diverse noise patterns for training models under different conditions.

## Boundary Condtions

- Implements boundary conditions for a 1D array with optional noise addition.  
- Takes input array `u`, grid spacing, number of interior cells `Ncell`, boundary condition mode, noise level, and an optional flag to return the noise-free version.  
- Adds noise to boundary cells using the `generate_noise` function.  
- Periodic condition matches boundaries to opposite edges of the array.  
- Returns the updated array and optionally the version without noise to store both noisy and noise free versions in the dataset.

Periodic boundary conditions connect the boundaries of a system so that values at one edge match those at the opposite edge, creating a continuous, looped domain. This is useful for simulating systems with cyclic or infinite behavior. In the Burgers' equation, periodic boundary conditions are often applied to study wave propagation, shock formation, and turbulence in a confined, repeating space, ensuring smooth transitions at the domain edges without artificial boundary effects.

## Van Leer limiter function

This code applies a **Van Leer limiter** to reconstruct left (`uL`) and right (`uR`) states of a solution variable `u` for finite volume methods. The `VLlimiter` limits gradients (`gradu`) between cells to ensure stability and avoid oscillations near discontinuities. These reconstructed states are used to compute numerical fluxes in PDE solvers, particularly for hyperbolic or conservation-law equations, enabling stable and accurate updates of the solution.

## Parameters

- `dt_save`: Time interval for saving simulation results.  
- `ini_time`: Initial time of the simulation.  
- `fin_time`: Final time of the simulation.  
- `nx`: Number of grid points in the spatial domain.  
- `xL`: Left boundary of the spatial domain.  
- `xR`: Right boundary of the spatial domain.  
- `if_second_order`: Flag to enable or disable second-order accuracy in calculations.  
- `show_steps`: Frequency of displaying or outputting simulation results.  

## Spatial and Temporal Grids

This code computes spatial and temporal grids for a numerical simulation:

- `dx`: Calculates the spatial resolution (grid spacing) as the total domain length divided by the number of grid points (`nx`).  
- `xe`: Generates the **edge coordinates** of the grid points using `jnp.linspace`, creating `nx + 1` evenly spaced points between `xL` and `xR`.  
- `xc`: Computes the **center coordinates** of each grid cell by averaging adjacent edge coordinates (`xe[:-1] + 0.5 * dx`).  
- `it_tot`: Determines the total number of time steps to save results, based on the total simulation time and saving interval (`dt_save`).  
- `tc`: Creates the time array for saved outputs, with `it_tot + 1` evenly spaced time values starting from 0.

## Hierarchial Storage

HDF5 (`h5py`) is used here for efficiently storing and organizing simulation data in a structured and portable format. The reasons for using HDF5 in this case are:

- **Hierarchical Organization**: The `coords` group organizes data logically, making it easy to manage related datasets (e.g., `x-coordinates` and `t-coordinates`).  
- **Scalability**: HDF5 handles large datasets efficiently, which is useful for high-resolution simulations like this one (`nx = 1024`, potentially large time arrays).  

The spatial (`xc`) and temporal (`tc`) coordinates are saved for later use or analysis in a structured and accessible way.

## **Dataset Generation function**  

The main `gen` function generates simulation data for a PDE with configurable physical parameters and noise settings. It initializes the solution with or without noise, evolves the system over time for both cases, and stores the results (e.g., solution states, boundary conditions, and parameters) in an HDF5 file. This enables efficient storage, reproducibility, and analysis of clean and noisy simulations for future training. 

### Inner Functions

- **`evolve`**: This function handles the temporal evolution of the solution array `u` by repeatedly applying the time-stepping logic in `simulation_fn` until the final simulation time is reached. It saves intermediate results at specified time intervals (`dt_save`) and handles clean and noisy simulations separately by incorporating noise levels. The function ensures efficient looping using `jax.lax.while_loop` for compatibility with JAX's just-in-time (JIT) compilation.

- **`simulation_fn`**: This function manages a single time step of the simulation. It calculates the time step size (`dt`) based on CFL constraints for advection and diffusion processes, applies the `update` function to compute the next state of `u`, and updates the time and step counters. It ensures the stability and consistency of the numerical solution through careful time-stepping.

- **`update`**: This function applies a two-stage time-stepping scheme to update the solution array `u`. It computes numerical fluxes using the `flux` function and incorporates noise at each stage if specified. The result is a stable numerical integration of the PDE over one time step, accounting for noise in both the flux and the governing equation.

- **`flux`**: This function calculates the fluxes needed to solve the PDE using an upwind scheme for stability and accuracy. It applies boundary conditions via the `bc` function and uses slope limiting with the `limiting` function to prevent unphysical oscillations. Additionally, it includes a source term for diffusion and optional noise to simulate variability in the flux.

- **`bc`**: This function defines the boundary conditions for the solution array `u`, such as periodic or reflective boundaries, and optionally adds noise to the boundary values. It ensures that the boundaries are correctly set for accurate flux calculations and consistent evolution of the PDE.

100 samples are generated for future training.

## Visualisation

The `visualize_burgers` function generates an animated GIF of the Burgers equation's solution over time. It takes spatial coordinates (`xcrd`), the simulation data (`data`), and an identifier (`i`) to name the output GIF. The function iterates through the time steps of the solution, plots each one, and stores the frames for the animation. It then creates an animation using `matplotlib.animation.ArtistAnimation`, saving it as a `.gif` file with a specified frame rate.

## Visualisation
# **Physics Informed Neural Networks**

Physics-Informed Neural Networks (PINNs) are a class of machine learning models that integrate the physical laws governing a system, typically in the form of partial differential equations (PDEs), into the neural network's training process. This approach allows the network to learn solutions to PDEs without needing a large dataset of solution points. Instead, the PINN enforces the PDEs as constraints in the loss function, guiding the network to discover solutions that satisfy the underlying physics. This makes PINNs particularly useful for solving complex problems in fluid dynamics, heat transfer, and other fields where data may be sparse or hard to obtain. 


## Model

This notebook defines a simple PINN model using TensorFlow. The class `PINN` constructs a fully connected neural network with layers specified by `layers_dims`. Each layer uses the `tanh` activation function, except the output layer, which produces a single continuous value. The model is designed to take inputs (such as spatial and time coordinates) and return a prediction, which can be further trained to satisfy the given physical laws expressed as PDEs.

## Physics informed loss

A function to compute the loss for training a Physics-Informed Neural Network (PINN) to solve a partial differential equation (PDE), while enforcing initial conditions, boundary conditions, and the PDE itself.
- **Operations** : 
  1. **Gradient Calculation**: 
     - Computes the first and second derivatives of the model's output (`u`) with respect to spatial (`x`) and time (`t`) using `tf.GradientTape`.
  2. **PDE Residual**: 
     - Calculates the residual of the PDE, involving terms like `u_t`, `u_x`, and `u_xx`, which should ideally be zero for a valid solution.

- **Loss Components**:
  - **Initial Condition Loss**: 
    - Measures the difference between the model's predicted solution at `t=0` and the given `initial_condition`.
  - **Periodic Boundary Condition Loss**: 
    - Enforces periodic boundary conditions by ensuring the solution and its spatial derivative are the same at the boundary points (`x_min` and `x_max`).
  - **Residual Loss**: 
    - Penalizes deviations from the PDE's residual to ensure the model satisfies the physical equation.

- **Output**:
  - Returns the total loss, which is the sum of the initial condition loss, periodic boundary loss, and residual loss.

# Training Loop


- **Operations**:
  1. **Grid Creation**: 
     - Creates a mesh grid of spatial (`x_grid`) and time (`t_grid`) values, then flattens them into 1D arrays (`x_flat` and `t_flat`).
  2. **Optimizer Setup**: 
     - Uses the Adam optimizer with the specified `learning_rate` to minimize the loss.
  3. **Training Loop**: 
     - For each epoch, computes the loss using `physics_informed_loss` and updates the model parameters by computing gradients and applying them through the optimizer.

## Data Loading


Opens the HDF5 file and retrieves:
- `initial_condition` and `boundary_condition`: Initial and boundary values for the solution, clean or noisy depending on input.
- `clean_data`: The ground truth solution data.
- `du`: Derivative of the solution.
- `epsilon`: A parameter (e.g., diffusion coefficient).
- `u0`: Initial value of the solution.
- `x` and `t`: Spatial and time coordinates.
  

## Model Setup

This code loads the simulation data from an HDF5 file and prepares it for training a Physics-Informed Neural Network (PINN) to solve the noisy version of the Burgers' equation.

  1. Load data from the file `simulation_data.h5` using the `load_data` function (loading initial conditions, boundary conditions, solution data, and other parameters).
  2. Convert the `x` and `t` coordinate arrays to TensorFlow tensors and reshapes them as column vectors.
  3. Convert `initial_condition` and `boundary_condition` to TensorFlow tensors.
  
- **Model Setup**: 
  - The PINN architecture is initialized with 3 hidden layers, each with 50 neurons (`layers_dims = [50, 50, 50]`).

## Training Noisy

## Prediction

#### Boundary Condition enforcement

This function enforces periodic boundary conditions on the solution array `u`.

  - Uses `tf.tensor_scatter_nd_update` to swap the values at the first and last indices of the solution array `u`, ensuring periodicity.

- Returns the modified solution array `u` with periodic boundary conditions applied, preserving the original shape.

#### Iterative time step prediction

  1. Initialize the solution with the `initial_condition` and store the first prediction.
  2. For each time step, concatenates the current spatial coordinates, solution, and `epsilon` to form the input for the model.
  3. The model predicts the solution at the next time step (`u_next`), and periodic boundary conditions are applied using the `apply_periodic_bc` function.
  4. Appends each new prediction to the list `all_predictions`.


#### Predictions 

Prepare the data and make predictions over loaded clean data

## Visualisation of Solution

  1. For each time step, it plots the spatial coordinates (`xcrd`) versus the solution at that step and stores the plot in `ims` for animation.
  2. Uses `ArtistAnimation` to create the animation from the collected frames.
  
  3. Saves the animation as a GIF using `PillowWriter` with a frame rate of 15 fps, and stores it at the specified `path`. The function also closes the figure after saving.

## Training Clean

Training over clean data for comparison

#### Prediction over clean data

#### Visualization# Kolmogorov Arnold Networks

### Kolmogorov-Arnold Networks (KANs): Overview  
- **Theoretical Foundation**: Based on the Kolmogorov-Arnold representation theorem, which states that any multivariate continuous function can be expressed as a sum of univariate functions, allowing for a systematic decomposition of complex functions.  
- **Architecture**: Decomposes high-dimensional mappings into a series of simpler one-dimensional functions, enabling efficient and accurate approximation of intricate dependencies in high-dimensional problems.  
- **Applications**: Particularly useful in fields requiring complex function approximations, such as machine learning, physics, and computational mathematics.

### KANs for Noise Removal in Partial Differential Equations (PDEs)  
- **Denoising Mechanism**: KANs excel at separating structured signals from stochastic noise by learning the underlying deterministic relationships within the data governed by PDEs.  
- **Training Process**: Trains to approximate noiseless PDE solutions by leveraging the network's bias towards representing smooth, well-structured functions.  
- **Advantage**: Provides robust and efficient recovery of clean PDE solutions from noisy datasets, combining theoretical rigor with practical performance.  
- **Practical Benefit**: Enhances the accuracy and reliability of numerical solutions for PDEs in scenarios where data is corrupted by noise.

## Implementation

This implementation of KAN has been taken from https://github.com/Blealtan/efficient-kan. 

This code defines a **Kolmogorov-Arnold Network (KAN) Linear Layer** in PyTorch. It is designed to efficiently model complex relationships by combining linear transformations and B-spline interpolation, as informed by the Kolmogorov-Arnold representation theorem.

###  Components:
1. **Initialization and Grid Setup**:
   - The grid represents the domain over which splines are interpolated. It is extended by the spline order to ensure smooth boundary conditions.
   - Parameters for the layer include base weights (`base_weight`) for linear transformations and spline weights (`spline_weight`) for B-spline coefficients.

2. **B-Spline Basis Calculation**:
   - The `b_splines` function computes B-spline bases for given inputs, which are then used to interpolate values in the input space.

3. **Spline Coefficient Computation**:
   - The `curve2coeff` function fits spline coefficients to the data by solving a least-squares problem, ensuring that the spline interpolation matches the input-output relationship.

4. **Forward Pass**:
   - Combines the linear transformation of the input using `base_weight` and the spline interpolation using `spline_weight` to produce the output.

5. **Grid Adaptation**:
   - The `update_grid` method adjusts the spline grid based on input distribution, allowing the model to adapt dynamically to new data.

6. **Regularization**:
   - A custom loss (`regularization_loss`) penalizes the spline weights to control overfitting and ensure smooth approximations. This includes terms for weight sparsity and entropy regularization.

This implementation is designed to be flexible and efficient, enabling it to approximate and learn high-dimensional functions, especially useful in tasks like function approximation, noise filtering, or PDE solutions.



## Layers

## Complete Model

- Defines a complete Kolmogorov-Arnold Network (KAN) as a sequence of KANLinear layers for multivariate function approximation.  
- Accepts hyperparameters like grid size, spline order, scaling factors, activation functions, and grid range to configure the architecture.  
- Uses a `ModuleList` to stack multiple KANLinear layers based on the input hidden layer configuration (`layers_hidden`).  
- Implements a forward pass that processes input through each KANLinear layer, optionally updating the grid dynamically for adaptability.  
- Provides a method to compute regularization loss by aggregating the regularization terms from all layers to promote sparsity and smoothness.

### Regularisation

- Extends the KAN architecture by integrating dropout layers to improve regularization and prevent overfitting.  
- Uses a `ModuleList` to stack layers, where each layer includes a KAN module, batch normalization, SiLU activation, and dropout.  
- Configurable dropout probability allows control over the level of regularization applied during training.  
- Implements a forward pass that processes input through each stacked layer sequentially, applying all operations in the layer pipeline.

## Custom Loss Function

- A custom loss function combining Mean Squared Error (MSE) and L1 loss for balanced optimization between precision and robustness.  
- The weighting factor `alpha` controls the contribution of each loss term, allowing flexibility based on the problem's requirements.  
- Uses the `forward` method to compute the weighted sum of MSE and L1 losses between the model's output and the target values.  
- Provides a smooth and robust loss function useful for tasks where both small errors and outlier handling are important.

# It has 1000 samples, push it as much as you can please till it runs out of VRAM

# modify for loop

## Data Loading

The clean and noisy samples of the solution are loaded for training

## Data Preparation

This script processes noisy and clean data for training a model, preparing it for PyTorch's DataLoader.

- **Data Reshaping**:
  - The noisy and clean datasets are flattened from `(num_samples, height, width)` into `(num_samples, height × width)` for compatibility with PyTorch models.

- **Tensor Conversion**:
  - The flattened arrays are converted into PyTorch tensors (`X_tensor` for noisy data and `Y_tensor` for clean data) of type `float32`.

- **Dataset Creation**:
  - A `TensorDataset` pairs the noisy data (`X_tensor`) with the clean target data (`Y_tensor`) for supervised learning.

- **Train/Test Split**:
  - The dataset is split into training and testing subsets, with 80% for training and 20% for testing, using PyTorch's `random_split`.

- **DataLoaders**:
  - Training and testing datasets are wrapped into DataLoaders with a batch size of 2, enabling shuffled training and sequential testing.

## Model Definition

  - A `KANWithDropout` model is initialized with the input dimension (`input_dim`), two hidden layers of sizes 256 and 64, and a final output layer of size `input_dim` to match the input/output shape.
  - The model is moved to the appropriate device (`GPU` if available, otherwise `CPU`).

- **Optimizer**:
  - Uses the `AdamW` optimizer, which combines the benefits of Adam with weight decay regularization to reduce overfitting.  
  - Learning rate is set to `1e-3`, and weight decay to `1e-4`.

- **Learning Rate Scheduler**:
  - A `ReduceLROnPlateau` scheduler reduces the learning rate by a factor of 0.5 when the monitored metric (e.g., loss) stops improving for 4 epochs (`patience=4`).
  - Minimum learning rate (`min_lr`) is set to `1e-6`, and verbose mode is enabled for logging changes.

- **Loss Function**:
  - Combines MSE and L1 losses using the `CombinedLoss` class, balancing precision and robustness during optimization.

## Training Loop

This code trains the `KANWithDropout` model for 200 epochs, evaluates its validation loss, adjusts the learning rate using a scheduler, and saves the trained model.

- **Training Loop**:
  - Iterates over 200 epochs.
  - For each batch:
    - Moves data (`inputs` and `targets`) to the appropriate device.
    - Clears previous gradients using `optimizer.zero_grad()`.
    - Computes model outputs, loss, and gradients, and updates weights using `optimizer.step()`.
    - Performs validation


- **Learning Rate Adjustment**:
  - The learning rate scheduler (`ReduceLROnPlateau`) adjusts the learning rate based on the validation loss, reducing it when the loss plateaus.

- **Model Saving**:
  - Saves the trained model's state dictionary (`model.state_dict()`) to a file named `"kan_model.pth"`, allowing for later use or deployment.


## Prediction

Randomly selected sample is reshaped and prepared to be fed to the model.

### Run Predictions

Predictions are made and the output is reshaped for visualisation.

## Visualisation

The `visualize_burgers` function generates an animated GIF of the Burgers equation's solution over time. It takes spatial coordinates (`xcrd`), the simulation data (`data`), and an identifier (`i`) to name the output GIF. The function iterates through the time steps of the solution, plots each one, and stores the frames for the animation. It then creates an animation using `matplotlib.animation.ArtistAnimation`, saving it as a `.gif` file with a specified frame rate.