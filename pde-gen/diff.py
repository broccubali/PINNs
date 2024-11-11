import numpy as np

s = np.load("data/burgerNoisy.npy")
t = np.load("data/burgerClean.npy")

print(np.linalg.norm(s - t))