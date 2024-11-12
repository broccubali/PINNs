from __future__ import annotations

import time
from math import ceil
import numpy as np
import jax
import jax.numpy as jnp
from jax import device_put, lax

# Configuration values
save_path = "data/"
dt_save = 0.01
ini_time = 0.0
fin_time = 1.0
nx = 1024
xL = 0.0
xR = 6.28318530718
nu = 1.0
rho = 1.0
CFL = 4.0e-1
if_show = 1
show_steps = 100
init_mode = "react"
noise_level = 0.1  # Noise level for initial condition, boundary condition, and equation


def add_gaussian_noise(u, mean=0.0, std=0.1):
    noise = np.random.normal(mean, std, size=u.shape)
    return u + noise


def Courant_diff(dx, epsilon=1.0e-3):
    stability_dif = 0.5 * dx**2 / (epsilon + 1.0e-8)
    return stability_dif


def bc(u, dx, Ncell, mode="periodic"):
    _u = jnp.zeros(Ncell + 4)  # because of 2nd-order precision in space
    _u = _u.at[2 : Ncell + 2].set(u)
    if mode == "periodic":  # periodic boundary condition
        _u = _u.at[0:2].set(u[-2:])  # left hand side
        _u = _u.at[Ncell + 2 : Ncell + 4].set(u[0:2])  # right hand side
    elif mode == "reflection":
        _u = _u.at[0].set(-u[3])  # left hand side
        _u = _u.at[1].set(-u[2])  # left hand side
        _u = _u.at[-2].set(-u[-3])  # right hand side
        _u = _u.at[-1].set(-u[-4])  # right hand side
    elif mode == "copy":
        _u = _u.at[0].set(u[3])  # left hand side
        _u = _u.at[1].set(u[2])  # left hand side
        _u = _u.at[-2].set(u[-3])  # right hand side
        _u = _u.at[-1].set(u[-4])  # right hand side

    # Add noise to the boundary condition
    # _u = add_gaussian_noise(_u, mean=0.0, std=noise_level)
    return _u


def init(xc, mode="sin", u0=1.0, du=0.1):
    """
    :param xc: cell center coordinate
    :param mode: initial condition
    :return: 1D scalar function u at cell center
    """
    modes = ["sin", "sinsin", "Gaussian", "react", "possin"]
    assert mode in modes, "mode is not defined!!"
    if mode == "sin":  # sinusoidal wave
        u = u0 * jnp.sin((xc + 1.0) * jnp.pi)
    elif mode == "sinsin":  # sinusoidal wave
        u = jnp.sin((xc + 1.0) * jnp.pi) + du * jnp.sin((xc + 1.0) * jnp.pi * 8.0)
    elif mode == "Gaussian":  # for diffusion check
        t0 = 0.01
        u = jnp.exp(-(xc**2) * jnp.pi / (4.0 * t0)) / jnp.sqrt(2.0 * t0)
    elif mode == "react":  # for reaction-diffusion eq.
        logu = -0.5 * (xc - jnp.pi) ** 2 / (0.25 * jnp.pi) ** 2
        u = jnp.exp(logu)
    elif mode == "possin":  # sinusoidal wave
        u = u0 * jnp.abs(jnp.sin((xc + 1.0) * jnp.pi))

    # Add noise to the initial condition
    u = add_gaussian_noise(u, mean=0.0, std=noise_level)
    return u


def main() -> None:
    print(f"nu: {nu:.3f}, rho: {rho:.3f}")

    # basic parameters
    dx = (xR - xL) / nx
    dx_inv = 1.0 / dx

    # cell edge coordinate
    xe = jnp.linspace(xL, xR, nx + 1)
    # cell center coordinate
    xc = xe[:-1] + 0.5 * dx
    # t-coordinate
    it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    tc = jnp.arange(it_tot + 1) * dt_save

    def evolve(u):
        t = ini_time
        tsave = t
        steps = 0
        i_save = 0
        tm_ini = time.time()
        dt = 0.0

        uu = jnp.zeros([it_tot, u.shape[0]])
        uu = uu.at[0].set(u)

        while t < fin_time:
            if t >= tsave:
                uu = uu.at[i_save].set(u)
                tsave += dt_save
                i_save += 1

            if steps % show_steps == 0 and if_show:
                print(f"now {steps:d}-steps, t = {t:.3f}, dt = {dt:.3f}")

            carry = (u, t, dt, steps, tsave)
            u, t, dt, steps, tsave = lax.fori_loop(0, show_steps, simulation_fn, carry)

            # Add noise to the equation
            u = add_gaussian_noise(u, mean=0.0, std=noise_level)

        tm_fin = time.time()
        print(f"total elapsed time is {tm_fin - tm_ini} sec")
        return uu, t

    @jax.jit
    def simulation_fn(i, carry):
        u, t, dt, steps, tsave = carry
        dt = Courant_diff(dx, nu) * CFL
        dt = jnp.min(jnp.array([dt, fin_time - t, tsave - t]))

        def _update(carry):
            u, dt = carry
            # preditor step for calculating t+dt/2-th time step
            u_tmp = update(u, u, dt * 0.5)
            # update using flux at t+dt/2-th time step
            u = update(u, u_tmp, dt)
            return u, dt

        def _pass(carry):
            return carry

        carry = u, dt
        u, dt = lax.cond(t > 1.0e-8, _update, _pass, carry)

        t += dt
        steps += 1
        return u, t, dt, steps, tsave

    @jax.jit
    def update(u, u_tmp, dt):
        # stiff part
        u = Piecewise_Exact_Solution(u, dt)
        # diffusion
        f = flux(u_tmp)
        u -= dt * dx_inv * (f[1 : nx + 1] - f[0:nx])
        return u

    @jax.jit
    def flux(u):
        _u = bc(u, dx, Ncell=nx)  # index 2 for _U is equivalent with index 0 for u
        # source term
        f = -nu * (_u[2 : nx + 3] - _u[1 : nx + 2]) * dx_inv
        return f

    @jax.jit
    def Piecewise_Exact_Solution(u, dt):  # Piecewise_Exact_Solution method
        # stiff equation
        u = 1.0 / (1.0 + jnp.exp(-rho * dt) * (1.0 - u) / u)
        return u

    u = init(xc=xc, mode=init_mode)
    u = device_put(u)  # putting variables in GPU (not necessary??)
    uu, t = evolve(u)
    print(f"final time is: {t:.3f}")

    print("data saving...")
    jnp.save(
        save_path + "ReacDiffNoisy",
        uu,
    )
    jnp.save(save_path + "x_coordinate_diff", xc)
    jnp.save(save_path + "t_coordinate_diff", tc)


if __name__ == "__main__":
    main()
