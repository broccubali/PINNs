from __future__ import annotations

import time
from math import ceil
import numpy as np
import jax
import jax.numpy as jnp
from jax import device_put
from scipy.stats import skewnorm

# Configuration values
save_path = ""
dt_save = 0.01
ini_time = 0.0
fin_time = 2.0
nx = 1024
xL = 0.0
xR = 1.0
beta = 1.0
if_show = 1
init_mode = "sin"
noise_level = 0.1  # Scale of noise
skewness = 1  # Skewness of the noise


def add_combined_noise(u, skewness=0.5, skew_scale=0.1, exp_scale=0.1, exp_mean=0.0):
    skew_noise = skewnorm.rvs(a=skewness, loc=0, scale=skew_scale, size=u.shape)
    exp_noise = np.random.exponential(scale=exp_scale, size=u.shape) + exp_mean
    combined_noise = skew_noise + exp_noise
    return u + combined_noise

def main() -> None:
    print(f"advection velocity: {beta}")

    # cell edge coordinate
    xe = jnp.linspace(xL, xR, nx + 1)
    # cell center coordinate
    xc = xe[:-1] + 0.5 * (xe[1] - xe[0])
    # t-coordinate
    it_tot = ceil((fin_time - ini_time) / dt_save) + 1
    tc = jnp.arange(it_tot + 1) * dt_save

    def evolve(u):
        t = ini_time
        i_save = 0
        tm_ini = time.time()

        it_tot = ceil((fin_time - ini_time) / dt_save) + 1
        uu = jnp.zeros([it_tot, u.shape[0]])
        uu = uu.at[0].set(u)

        while t < fin_time:
            print(f"save data at t = {t:.3f}")
            u = set_function(xc, t, beta)
            u = add_combined_noise(u, skewness=-1, skew_scale=0.8, exp_scale=0.09, exp_mean=0.0)
            uu = uu.at[i_save].set(u)
            t += dt_save
            i_save += 1

        tm_fin = time.time()
        print(f"total elapsed time is {tm_fin - tm_ini} sec")
        uu = uu.at[-1].set(u)
        return uu, t

    @jax.jit
    def set_function(x, t, beta):
        u = jnp.sin(2.0 * jnp.pi * (x - beta * t))
        u = add_combined_noise(u, skewness=-2, skew_scale=0.1, exp_scale=0.09, exp_mean=0.0)
        return u

    u = seNt_function(xc, t=0, beta=beta)
    u = device_put(u)  # putting variables in GPU (not necessary??)
    uu, t = evolve(u)
    print(f"final time is: {t:.3f}")

    print("data saving...")
    jnp.save(save_path + "pde-gen/advection/data/AdvectionCombo.npy", uu)
    jnp.save(save_path + "pde-gen/advection/data/x_coordinate_adv.npy", xe)
    jnp.save(save_path + "pde-gen/advection/data/t_coordinate_adv.npy", tc)


if __name__ == "__main__":
    main()
