from __future__ import annotations

import random
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
from jax import device_put, lax
import numpy as np
from math import ceil


def _pass(carry):
    return carry


# Configuration values
save_path = "data/"
dt_save = 0.01
ini_time = 0.0
fin_time = 1.0
nx = 1024
ny = 1024
xL = 0.0
xR = 6.28318530718
yL = 0.0
yR = 6.28318530718
beta_x = 1.0
beta_y = 1.0
CFL = 4.0e-1
if_show = 1
show_steps = 100
init_mode = "react"
noise_level = 0.1  # Noise level for initial condition, boundary condition, and equation


def Courant_2d(beta, dx, dy):
    return jnp.min(jnp.array([dx / jnp.abs(beta[0]), dy / jnp.abs(beta[1])]))


def bc_2d(u, dx, dy, nx, ny, mode="periodic"):
    _u = jnp.zeros((nx + 4, ny + 4))  # because of 2nd-order precision in space
    _u = _u.at[2 : nx + 2, 2 : ny + 2].set(u)
    if mode == "periodic":  # periodic boundary condition
        _u = _u.at[0:2, 2 : ny + 2].set(u[-2:, :])  # left hand side
        _u = _u.at[nx + 2 : nx + 4, 2 : ny + 2].set(u[0:2, :])  # right hand side
        _u = _u.at[2 : nx + 2, 0:2].set(u[:, -2:])  # bottom side
        _u = _u.at[2 : nx + 2, ny + 2 : ny + 4].set(u[:, 0:2])  # top side
        _u = _u.at[0:2, 0:2].set(u[-2:, -2:])  # bottom-left corner
        _u = _u.at[nx + 2 : nx + 4, 0:2].set(u[0:2, -2:])  # bottom-right corner
        _u = _u.at[0:2, ny + 2 : ny + 4].set(u[-2:, 0:2])  # top-left corner
        _u = _u.at[nx + 2 : nx + 4, ny + 2 : ny + 4].set(
            u[0:2, 0:2]
        )  # top-right corner
    elif mode == "reflection":
        _u = _u.at[0:2, 2 : ny + 2].set(-u[3:1:-1, :])  # left hand side
        _u = _u.at[nx + 2 : nx + 4, 2 : ny + 2].set(-u[-4:-2, :])  # right hand side
        _u = _u.at[2 : nx + 2, 0:2].set(-u[:, 3:1:-1])  # bottom side
        _u = _u.at[2 : nx + 2, ny + 2 : ny + 4].set(-u[:, -4:-2])  # top side
        _u = _u.at[0:2, 0:2].set(-u[3:1:-1, 3:1:-1])  # bottom-left corner
        _u = _u.at[nx + 2 : nx + 4, 0:2].set(-u[-4:-2, 3:1:-1])  # bottom-right corner
        _u = _u.at[0:2, ny + 2 : ny + 4].set(-u[3:1:-1, -4:-2])  # top-left corner
        _u = _u.at[nx + 2 : nx + 4, ny + 2 : ny + 4].set(
            -u[-4:-2, -4:-2]
        )  # top-right corner
    elif mode == "copy":
        _u = _u.at[0:2, 2 : ny + 2].set(u[3:1:-1, :])  # left hand side
        _u = _u.at[nx + 2 : nx + 4, 2 : ny + 2].set(u[-4:-2, :])  # right hand side
        _u = _u.at[2 : nx + 2, 0:2].set(u[:, 3:1:-1])  # bottom side
        _u = _u.at[2 : nx + 2, ny + 2 : ny + 4].set(u[:, -4:-2])  # top side
        _u = _u.at[0:2, 0:2].set(u[3:1:-1, 3:1:-1])  # bottom-left corner
        _u = _u.at[nx + 2 : nx + 4, 0:2].set(u[-4:-2, 3:1:-1])  # bottom-right corner
        _u = _u.at[0:2, ny + 2 : ny + 4].set(u[3:1:-1, -4:-2])  # top-left corner
        _u = _u.at[nx + 2 : nx + 4, ny + 2 : ny + 4].set(
            u[-4:-2, -4:-2]
        )  # top-right corner

    return _u

def limiting_2d(u, nx, ny, axis=0, if_second_order=True):
    if if_second_order:
        if axis == 0:
            uL = u[:-1, :]
            uR = u[1:, :]
        elif axis == 1:
            uL = u[:, :-1]
            uR = u[:, 1:]
    else:
        uL = u
        uR = u
    return uL, uR

def init_multi_2d(xc, yc, numbers=1, k_tot=4, init_key="sin"):
    """
    :param xc: cell center coordinate in x direction
    :param yc: cell center coordinate in y direction
    :param numbers: number of initial conditions
    :param k_tot: number of modes
    :param init_key: initial condition key
    :return: 2D scalar function u at cell center
    """
    u = jnp.zeros((numbers, xc.shape[0], yc.shape[0]))
    for i in range(numbers):
        if init_key == "sin":
            u = u.at[i].set(jnp.sin(k_tot * xc[:, None]) * jnp.sin(k_tot * yc[None, :]))
        elif init_key == "gaussian":
            u = u.at[i].set(
                jnp.exp(
                    -((xc[:, None] - 0.5) ** 2 + (yc[None, :] - 0.5) ** 2) / (0.1**2)
                )
            )
    return u

def main() -> None:
    print(f"beta_x: {beta_x:.3f}, beta_y: {beta_y:.3f}")

    # basic parameters
    dx = (xR - xL) / nx
    dy = (yR - yL) / ny
    dx_inv = 1.0 / dx
    dy_inv = 1.0 / dy

    # cell edge coordinate
    xe = jnp.linspace(xL, xR, nx + 1)
    ye = jnp.linspace(yL, yR, ny + 1)
    # cell center coordinate
    xc = xe[:-1] + 0.5 * dx
    yc = ye[:-1] + 0.5 * dy
    # t-coordinate
    it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    tc = jnp.arange(it_tot + 1) * dt_save

    def evolve(u):
        t = ini_time
        tsave = t
        steps = 0
        i_save = 0
        dt = 0.0
        uu = jnp.zeros([it_tot, u.shape[1], u.shape[2]])  # Adjusted shape
        uu = uu.at[0].set(u[0])  # Set the initial condition correctly

        cond_fun = lambda x: x[0] < fin_time

        def _body_fun(carry):
            def _show(_carry):
                u, tsave, i_save, uu = _carry
                uu = uu.at[i_save].set(u[0])  # Set the current state correctly
                tsave += dt_save
                i_save += 1
                return (u, tsave, i_save, uu)

            t, tsave, steps, i_save, dt, u, uu = carry

            carry = (u, tsave, i_save, uu)
            u, tsave, i_save, uu = lax.cond(t >= tsave, _show, _pass, carry)

            carry = (u, t, dt, steps, tsave)
            u, t, dt, steps, tsave = lax.fori_loop(0, show_steps, simulation_fn, carry)

            return (t, tsave, steps, i_save, dt, u, uu)

        carry = t, tsave, steps, i_save, dt, u, uu
        t, tsave, steps, i_save, dt, u, uu = lax.while_loop(cond_fun, _body_fun, carry)
        uu = uu.at[-1].set(u[0])  # Set the final state correctly

        return uu

    @jax.jit
    def simulation_fn(i, carry):
        u, t, dt, steps, tsave = carry
        dt = Courant_2d(jnp.array([beta_x, beta_y]), dx, dy) * CFL
        dt = jnp.min(jnp.array([dt, fin_time - t, tsave - t]))

        def _update(carry):
            u, dt = carry
            # preditor step for calculating t+dt/2-th time step
            u_tmp = update(u, u, dt * 0.5)
            # update using flux at t+dt/2-th time step
            u = update(u, u_tmp, dt)
            return u, dt

        carry = u, dt
        u, dt = lax.cond(dt > 1.0e-8, _update, _pass, carry)

        t += dt
        steps += 1
        return u, t, dt, steps, tsave

    @jax.jit
    def update(u, u_tmp, dt):
        f_x, f_y = flux(u_tmp)
        u -= dt * (
            dx_inv * (f_x[1 : nx + 1, :] - f_x[0:nx, :])
            + dy_inv * (f_y[:, 1 : ny + 1] - f_y[:, 0:ny])
        )
        return u

    def flux(u):
        _u = bc_2d(u[0], dx, dy, nx, ny)  # index 2 for _U is equivalent with index 0 for u
        uL_x, uR_x = limiting_2d(_u, nx, ny, axis=0, if_second_order=True)
        uL_y, uR_y = limiting_2d(_u, nx, ny, axis=1, if_second_order=True)
        fL_x = uL_x * beta_x
        fR_x = uR_x * beta_x
        fL_y = uL_y * beta_y
        fR_y = uR_y * beta_y
        # upwind advection scheme
        f_upwd_x = 0.5 * (
            fR_x[1 : nx + 2, :]
            + fL_x[2 : nx + 3, :]
            - jnp.abs(beta_x) * (uL_x[2 : nx + 3, :] - uR_x[1 : nx + 2, :])
        )
        f_upwd_y = 0.5 * (
            fR_y[:, 1 : ny + 2]
            + fL_y[:, 2 : ny + 3]
            - jnp.abs(beta_y) * (uL_y[:, 2 : ny + 3] - uR_y[:, 1 : ny + 2])
        )
        return f_upwd_x, f_upwd_y

    u = init_multi_2d(xc, yc, numbers=1, k_tot=4, init_key=init_mode)
    u = device_put(u)  # putting variables in GPU (not necessary??)

    uu = evolve(u)

    print("data saving...")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    jnp.save(save_path + "advection2D", uu)
    jnp.save(save_path + "x_coordinate_adv2d", xc)
    jnp.save(save_path + "y_coordinate_adv2d", yc)
    jnp.save(save_path + "t_coordinate_adv2d", tc)


if __name__ == "__main__":
    main()