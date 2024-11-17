- Noise can appear in:

    - Initial/boundary conditions (dependent variable constraints).
    - Observational data for the solution.
    - Independent variables (coordinates like xx, tt).
    - Physical parameters in the PDE.


1. So data should contain

    - initial conditon
    - initial condition with noise
    - boundary condition
    - boundary condition with noise
    - parameters of the equation
    - parameters with noise
    - solution clean
    - solution with noise
    - noisy x??


2. Some of the data should also be completely clean. We need to teach the model to differentiate between clean and noisy solutions.

3. Feed initial, boundary conditions and parameters to PINN, generate broken solution.

4. Pass broken solution to KAN, teach it difference between normal and clean and voila.

5. S grade, journal paper.