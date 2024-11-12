from __future__ import annotations

import sys
import time
from math import ceil

import jax
import jax.numpy as jnp
from jax import device_put, lax

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")


def _pass(carry):
    return carry


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
    return u


def Courant(u, dx):
    stability_adv = dx / (jnp.max(jnp.abs(u)) + 1.0e-8)
    return stability_adv


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

    return _u


def VLlimiter(a, b, c, alpha=2.0):
    return (
        jnp.sign(c)
        * (0.5 + 0.5 * jnp.sign(a * b))
        * jnp.minimum(alpha * jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.abs(c))
    )


def limiting(u, Ncell, if_second_order):
    du_L = u[1 : Ncell + 3] - u[0 : Ncell + 2]
    du_R = u[2 : Ncell + 4] - u[1 : Ncell + 3]
    du_M = (u[2 : Ncell + 4] - u[0 : Ncell + 2]) * 0.5
    gradu = VLlimiter(du_L, du_R, du_M) * if_second_order
    uL, uR = jnp.zeros_like(u), jnp.zeros_like(u)
    uL = uL.at[1 : Ncell + 3].set(u[1 : Ncell + 3] - 0.5 * gradu)
    uR = uR.at[1 : Ncell + 3].set(u[1 : Ncell + 3] + 0.5 * gradu)
    return uL, uR


def main() -> None:
    # Configuration values
    save_path = "/content"
    dt_save = 0.01
    ini_time = 0.0
    fin_time = 2.0
    nx = 1024
    xL = -1.0
    xR = 1.0
    epsilon = 1.0e-2
    u0 = 1.0
    du = 0.1
    CFL = 4.0e-1
    if_second_order = 1.0
    show_steps = 100
    init_mode = "sin"

    # Basic parameters
    pi_inv = 1.0 / jnp.pi
    dx = (xR - xL) / nx
    dx_inv = 1.0 / dx

    # Cell edge coordinate
    xe = jnp.linspace(xL, xR, nx + 1)
    # Cell center coordinate
    xc = xe[:-1] + 0.5 * dx

    # Time parameters
    it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    tc = jnp.arange(it_tot + 1) * dt_save

    @jax.jit
    def evolve(u):
        t = ini_time
        tsave = t
        steps = 0
        i_save = 0
        dt = 0.0
        uu = jnp.zeros([it_tot, u.shape[0]])
        uu = uu.at[0].set(u)

        tm_ini = time.time()

        cond_fun = lambda x: x[0] < fin_time

        def _body_fun(carry):
            def _save(_carry):
                u, tsave, i_save, uu = _carry
                uu = uu.at[i_save].set(u)
                tsave += dt_save
                i_save += 1
                return (u, tsave, i_save, uu)

            t, tsave, steps, i_save, dt, u, uu = carry

            # if save data
            carry = (u, tsave, i_save, uu)
            u, tsave, i_save, uu = lax.cond(t >= tsave, _save, _pass, carry)

            carry = (u, t, dt, steps, tsave)
            u, t, dt, steps, tsave = lax.fori_loop(0, show_steps, simulation_fn, carry)

            return (t, tsave, steps, i_save, dt, u, uu)

        carry = t, tsave, steps, i_save, dt, u, uu
        t, tsave, steps, i_save, dt, u, uu = lax.while_loop(cond_fun, _body_fun, carry)
        uu = uu.at[-1].set(u)

        tm_fin = time.time()
        print(f"total elapsed time is {tm_fin - tm_ini} sec")
        return uu, t

    @jax.jit
    def simulation_fn(i, carry):
        u, t, dt, steps, tsave = carry
        dt_adv = Courant(u, dx) * CFL
        dt_dif = Courant_diff(dx, epsilon * pi_inv) * CFL
        dt = jnp.min(jnp.array([dt_adv, dt_dif, fin_time - t, tsave - t]))

        def _update(carry):
            u, dt = carry
            # Predictor step for calculating t+dt/2-th time step
            u_tmp = update(u, u, dt * 0.5)
            # Update using flux at t+dt/2-th time step
            u = update(u, u_tmp, dt)
            return u, dt

        carry = u, dt
        u, dt = lax.cond(dt > 1.0e-8, _update, _pass, carry)

        t += dt
        steps += 1
        return u, t, dt, steps, tsave

    @jax.jit
    def update(u, u_tmp, dt):
        f = flux(u_tmp)
        u -= dt * dx_inv * (f[1 : nx + 1] - f[0:nx])
        return u

    def flux(u):
        _u = bc(u, dx, Ncell=nx)  # index 2 for _U is equivalent with index 0 for u
        uL, uR = limiting(_u, nx, if_second_order=if_second_order)
        fL = 0.5 * uL**2
        fR = 0.5 * uR**2
        # Upwind advection scheme
        f_upwd = 0.5 * (
            fR[1 : nx + 2]
            + fL[2 : nx + 3]
            - 0.5
            * jnp.abs(uL[2 : nx + 3] + uR[1 : nx + 2])
            * (uL[2 : nx + 3] - uR[1 : nx + 2])
        )
        # Source term
        f_upwd += -epsilon * pi_inv * (_u[2 : nx + 3] - _u[1 : nx + 2]) * dx_inv
        return f_upwd

    u = init(xc=xc, mode=init_mode, u0=u0, du=du)
    u = device_put(u)  # Putting variables in GPU (not necessary??)
    uu, t = evolve(u)
    print(f"final time is: {t:.3f}")

    print("data saving...")
    jnp.save("data/burgerClean", uu)
    jnp.save("data/x_coordinate", xc)
    jnp.save("data/t_coordinate", tc)


if __name__ == "__main__":
    main()
