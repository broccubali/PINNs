# Inverse-PINN with for channel flow
# Data is retrieved in get_data.py

# -*- coding: utf-8 -*-

# Tensorflow version: 1.14
# Python version: 3.7.*

# Disables warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import numpy as np
import sys
from   mod import *
from   pylab import *
from   dom import Run, implot
# LaTex
rc('font',**{'size':20})
rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{bm}')

tf.compat.v1.disable_eager_execution()

# Get parameters
which  = "{:02}".format(int(sys.argv[1]))
params = Run(which)

# -----------------------------------------------------------------------------
# Create the Neural Network 
# -----------------------------------------------------------------------------

# Here we create the NN, both the data part and the physics informed part. We
# also define the loss function and the optimizer, start the session, and
# initialize the variables.

# Shape of the NN
layers = [4]+[params.hu]*params.layers+[4]
L      = len(layers)

# Neurons
weights = ([xavier_init([layers[l], layers[l+1]])
            for l in range(0, L-1)])
biases  = ([tf.Variable(tf.zeros((1, layers[l+1]),dtype=tf.float64))
            for l in range(0, L-1)])
if params.adp:
    adp = tf.Variable(0.1, dtype=tf.float64)
else:
    adp = 0.1

# Points where the data contraints are going to be imposed
t_u = tf.compat.v1.placeholder(tf.float64, shape=(None,1))
x_u = tf.compat.v1.placeholder(tf.float64, shape=(None,1))
y_u = tf.compat.v1.placeholder(tf.float64, shape=(None,1))
z_u = tf.compat.v1.placeholder(tf.float64, shape=(None,1))
field_u,_ = DNN(tf.concat((t_u,x_u,y_u,z_u),axis=1),
                layers, weights, biases, act=params.act, adpt=adp)
u_u = field_u[:,0:1]
v_u = field_u[:,1:2]
w_u = field_u[:,2:3]
p_u = field_u[:,3:]

u_obs = tf.compat.v1.placeholder(tf.float64, shape=(None,1))
v_obs = tf.compat.v1.placeholder(tf.float64, shape=(None,1))
w_obs = tf.compat.v1.placeholder(tf.float64, shape=(None,1))

# Loss function for data constraints
loss_u = (tf.reduce_mean(tf.square(u_u-u_obs)) +
          tf.reduce_mean(tf.square(v_u-v_obs)) +
          tf.reduce_mean(tf.square(w_u-w_obs)))

# Points where the dynamical contraints are going to be imposed
t_f = tf.compat.v1.placeholder(tf.float64, shape=(None,1))
x_f = tf.compat.v1.placeholder(tf.float64, shape=(None,1))
y_f = tf.compat.v1.placeholder(tf.float64, shape=(None,1))
z_f = tf.compat.v1.placeholder(tf.float64, shape=(None,1))
field_f,_ = DNN(tf.concat((t_f,x_f,y_f,z_f),axis=1),
                layers, weights, biases, act=params.act, adpt=adp)
u_f = field_f[:,0:1]
v_f = field_f[:,1:2]
w_f = field_f[:,2:3]
p_f = field_f[:,3:]

# Automatic differentiation and eqs
u_t = tf.gradients(u_f, t_f)[0]
v_t = tf.gradients(v_f, t_f)[0]
w_t = tf.gradients(w_f, t_f)[0]

u_x = tf.gradients(u_f, x_f)[0]
u_y = tf.gradients(u_f, y_f)[0]
u_z = tf.gradients(u_f, z_f)[0]

v_x = tf.gradients(v_f, x_f)[0]
v_y = tf.gradients(v_f, y_f)[0]
v_z = tf.gradients(v_f, z_f)[0]

w_x = tf.gradients(w_f, x_f)[0]
w_y = tf.gradients(w_f, y_f)[0]
w_z = tf.gradients(w_f, z_f)[0]

p_x = tf.gradients(p_f, x_f)[0]
p_y = tf.gradients(p_f, y_f)[0]
p_z = tf.gradients(p_f, z_f)[0]

u_xx = tf.gradients(u_x, x_f)[0]
u_yy = tf.gradients(u_y, y_f)[0]
u_zz = tf.gradients(u_z, z_f)[0]

v_xx = tf.gradients(v_x, x_f)[0]
v_yy = tf.gradients(v_y, y_f)[0]
v_zz = tf.gradients(v_z, z_f)[0]

w_xx = tf.gradients(w_x, x_f)[0]
w_yy = tf.gradients(w_y, y_f)[0]
w_zz = tf.gradients(w_z, z_f)[0]

if params.pressure:
    # Laplacian of pressure
    p_xx = tf.gradients(p_x, x_f)[0]
    p_yy = tf.gradients(p_y, y_f)[0]
    p_zz = tf.gradients(p_z, z_f)[0]

    # Grad grad u_i u_j
    uu = u_f*u_f
    uv = u_f*v_f
    uw = u_f*w_f
    vv = v_f*v_f
    vw = v_f*w_f
    ww = w_f*w_f

    dduu = tf.gradients(uu,   x_f)[0]
    dduu = tf.gradients(dduu, x_f)[0]

    dduv = tf.gradients(uv,   x_f)[0]
    dduv = tf.gradients(dduv, y_f)[0]

    dduw = tf.gradients(uw,   x_f)[0]
    dduw = tf.gradients(dduw, z_f)[0]

    ddvv = tf.gradients(vv,   y_f)[0]
    ddvv = tf.gradients(ddvv, y_f)[0]

    ddvw = tf.gradients(vw,   y_f)[0]
    ddvw = tf.gradients(ddvw, z_f)[0]

    ddww = tf.gradients(ww,   z_f)[0]
    ddww = tf.gradients(ddww, z_f)[0]

    f0 = (p_xx+p_yy+p_zz) + (dduu + 2*dduv + 2*dduw
                                  +   ddvv + 2*ddvw
                                           +   ddww)
else:
    f0 = u_x+v_y+w_z

# Params eq
nu = 5e-5
dPdx = 0.0025

# Loss function for dynamical constraints
lfw = tf.Variable(params.lfw, trainable=False, dtype=tf.float64)
if   params.eqs=='momentum':
    f1 = (u_t + u_f*u_x + v_f*u_y + w_f*u_z +
            p_x + dPdx - nu*(u_xx+u_yy+u_zz))
    f2 = (v_t + u_f*v_x + v_f*v_y + w_f*v_z +
            p_y - nu*(v_xx+v_yy+v_zz))
    f3 = (w_t + u_f*w_x + v_f*w_y + w_f*w_z +
            p_z - nu*(w_xx+w_yy+w_zz))

    loss_f = lfw*(tf.reduce_mean(tf.square(f0)) +
                  tf.reduce_mean(tf.square(f1)) +
                  tf.reduce_mean(tf.square(f2)) +
                  tf.reduce_mean(tf.square(f3)))
elif params.eqs=='energy':
    eng   = 0.5*(u_f*u_f + v_f*v_f + w_f*w_f)
    eng_t = tf.gradients(eng, t_f)[0]
    trans = (u_f*(u_f*u_x + v_f*u_y + w_f*u_z) +
             v_f*(u_f*v_x + v_f*v_y + w_f*v_z) +
             w_f*(u_f*w_x + v_f*w_y + w_f*w_z))
    prtr = u_f*(p_x + dPdx) + v_f*p_y + w_f*p_z
    vstf = nu*(u_f*(u_xx+u_yy+u_zz) +
               v_f*(v_xx+v_yy+v_zz) +
               w_f*(w_xx+w_yy+w_zz))
    feng = eng_t + trans + prtr - vstf

    loss_f = lfw*(tf.reduce_mean(tf.square(feng)) +
                  tf.reduce_mean(tf.square(f0)))
if   params.eqs=='momentum':
    o_x = w_y - v_z
    o_y = u_z - w_x
    o_z = v_x - u_y

    o_xt = tf.gradients(o_x, t_f)[0]
    o_yt = tf.gradients(o_x, t_f)[0]
    o_zt = tf.gradients(o_x, t_f)[0]

    f1 = (u_t + u_f*u_x + v_f*u_y + w_f*u_z +
            p_x + dPdx - nu*(u_xx+u_yy+u_zz))
    f2 = (v_t + u_f*v_x + v_f*v_y + w_f*v_z +
            p_y - nu*(v_xx+v_yy+v_zz))
    f3 = (w_t + u_f*w_x + v_f*w_y + w_f*w_z +
            p_z - nu*(w_xx+w_yy+w_zz))

    loss_f = lfw*(tf.reduce_mean(tf.square(f0)) +
                  tf.reduce_mean(tf.square(f1)) +
                  tf.reduce_mean(tf.square(f2)) +
                  tf.reduce_mean(tf.square(f3)))

# Total loss function
loss = loss_f + loss_u

# Div condition
div_norm = tf.square(u_x) + tf.square(v_y) + tf.square(w_z)
div_cond = tf.square(u_x+v_y+w_z)/div_norm
div_cond = tf.reduce_mean(div_cond)

# Optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(5.0e-4).minimize(loss)

# Create save object
saver = tf.compat.v1.train.Saver()

# Run Tensorflow session
sess = tf.compat.v1.Session()

# Restore session 
saver.restore(sess, "{}/session".format(which))

# Points for plotting
Nx = 64
Ny = 8
Nz = 64
visc_length = 1.0006e-3
u_tau       = 4.9968e-2
dt          = 0.0065
dx          = 8*np.pi/2048
dz          = 3*np.pi/1536 
dx_plus     = dx/visc_length
dz_plus     = dz/visc_length
dt_plus     = dt*u_tau/visc_length
# for tidx in [0,70,149]:
for tidx in [0]:
    t_p, x_p, y_p, z_p = plot_points(Nx, Ny, Nz, tidx=tidx)

    # Save fields
    u_p, v_p, w_p, p_p = sess.run([u_u,v_u,w_u,p_u],
                                  feed_dict={t_u: t_p,
                                             x_u: x_p,
                                             y_u: y_p,
                                             z_u: z_p})
    div_cond_val = sess.run(div_cond, 
                                  feed_dict={t_u: t_p,
                                             x_u: x_p,
                                             y_u: y_p,
                                             z_u: z_p,
                                             t_f: t_p,
                                             x_f: x_p,
                                             y_f: y_p,
                                             z_f: z_p})
    print(div_cond_val)
    u_p = u_p.reshape(Nx,Ny,Nz)
    v_p = v_p.reshape(Nx,Ny,Nz)
    w_p = w_p.reshape(Nx,Ny,Nz)
    p_p = p_p.reshape(Nx,Ny,Nz)

    # Real fields
    vv = np.load("data/velos.{:02}.npy".format(tidx))
    pp = np.load("data/press.{:02}.npy".format(tidx))

    fig = figure(1, figsize=(20,10))

    subplot(241)
    implot(vv[0,:,5,:], extent=[0,dx_plus*64,0,dz_plus*64])
    ylabel('$z^+$')
    # xlabel('$x^+$')
    title('Real $u$, $t^+ = {:.0f}$'.format(tidx*dt_plus))
    # savefig(f"figs/real_u_{which}_{tidx:03}")
    subplot(242)
    implot(vv[1,:,5,:], extent=[0,dx_plus*64,0,dz_plus*64])
    title('Real $w$')
    # ylabel('$z^+$')
    # xlabel('$x^+$')
    title('Real $v$, $t^+ = {:.0f}$'.format(tidx*dt_plus))
    # savefig(f"figs/real_v_{which}_{tidx:03}")
    subplot(243)
    implot(vv[2,:,5,:], extent=[0,dx_plus*64,0,dz_plus*64])
    # ylabel('$z^+$')
    # xlabel('$x^+$')
    title('Real $w$, $t^+ = {:.0f}$'.format(tidx*dt_plus))
    # savefig(f"figs/real_w_{which}_{tidx:03}")
    subplot(244)
    implot(pp[0,:,5,:], extent=[0,dx_plus*64,0,dz_plus*64])
    # ylabel('$z^+$')
    # xlabel('$x^+$')
    title('Real $p$, $t^+ = {:.0f}$'.format(tidx*dt_plus))
    # savefig(f"figs/real_p_{which}_{tidx:03}")

    subplot(245)
    implot(u_p[:,5,:], extent=[0,dx_plus*64,0,dz_plus*64])
    ylabel('$z^+$')
    xlabel('$x^+$')
    title('PINN $u$, $t^+ = {:.0f}$'.format(tidx*dt_plus))
    # savefig(f"figs/pinn_u_{which}_{tidx:03}")
    subplot(246)
    implot(v_p[:,5,:], extent=[0,dx_plus*64,0,dz_plus*64])
    # ylabel('$z^+$')
    xlabel('$x^+$')
    title('PINN $v$, $t^+ = {:.0f}$'.format(tidx*dt_plus))
    # savefig(f"figs/pinn_v_{which}_{tidx:03}")
    subplot(247)
    implot(w_p[:,5,:], extent=[0,dx_plus*64,0,dz_plus*64])
    # ylabel('$z^+$')
    xlabel('$x^+$')
    title('PINN $w$, $t^+ = {:.0f}$'.format(tidx*dt_plus))
    # savefig(f"figs/pinn_w_{which}_{tidx:03}")
    subplot(248)
    implot(p_p[:,5,:], extent=[0,dx_plus*64,0,dz_plus*64])
    # ylabel('$z^+$')
    xlabel('$x^+$')
    title('PINN $p$, $t^+ = {:.0f}$'.format(tidx*dt_plus))
    # savefig(f"figs/pinn_p_{which}_{tidx:03}")

    savefig(f"figs/fields{which}_{tidx:03}")

# draw()
# show()
