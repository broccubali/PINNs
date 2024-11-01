import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import loadmat
# Load the data
a = np.loadtxt("/home/shusrith/Downloads/K_16.dat")

a = a.reshape(512, 201)
# a = a["uu"]
print(a.shape)
# Set up the figure, axis, and plot element
fig, ax = plt.subplots()
(line,) = ax.plot(a[:, 0])

# Set the axis limits
ax.set_xlim(0, a.shape[0])
ax.set_ylim(np.min(a), np.max(a))


# Animation function
def animate(i):
    line.set_ydata(a[:, i])
    return (line,)


# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=a.shape[1], interval=50, blit=True)

# Save the animation as a GIF
ani.save("pde_solution.gif", writer="imagemagick")
