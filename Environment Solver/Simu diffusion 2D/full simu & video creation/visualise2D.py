import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# import jet from matplotlib
from matplotlib.cm import jet

# Load data
data = np.load('decay_c_36x36x20000x4x4x2x0.4x1.0x5.0x-1.0x-1.0x1.0x25.0.npy') # decay_ or not

# Normalize the colormap based on the maximum and minimum values across all images
norm = Normalize(vmin=data.min(), vmax=data.max())

# Initialize model domains
Lx = 4     # Default 4     (Length of x domain)viri
Ly = 4     # Default 4     (Length of y domain)
Lt = 5     # Default 2     (Length of t domain)
Nx = 36    # Default 100   (Gridpoints in x)
Ny = 36    # Default 100   (Gridpoints in y)
Nt = 50000 # Default 20000 (Computational timesteps)
Ntp = 5000 # Default 2000  (Saved timesteps)

# Initialize model step sizes and grid
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = Lt / (Nt - 1)
x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)
timesteps = np.linspace(0, Lt, Ntp)
[X, Y] = np.meshgrid(x, y)

# Define a function to map normalized concentration values to point sizes
def scale_point_size(x):
    x = (x - x.min()) / (x.max() - x.min())
    min_size = 0  # Minimum point size
    max_size = 100  # Maximum point size
    return min_size + (max_size - min_size) * x

# Define a function to map normalized concentration values to alpha values
def alpha_values_fcn(C):
    C = (C - C.min()) / (C.max() - C.min())
    min_alpha = 0.1  # Minimum alpha value
    max_alpha = 1.0  # Maximum alpha value
    return min_alpha + (max_alpha - min_alpha) * C

# Iterate over saved timesteps
for i in range(0, len(data[0, 0, :]), 10):
    print(i)
    # Plot 2D concentration field
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(data[:, :, i], cmap=jet, norm=norm, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], origin='lower')
    plt.colorbar()
    plt.title(f'2D Concentration Field at t = {timesteps[i]}')
    plt.xlabel('x')
    fig.text(0.8, 0.02, i, fontsize=10)
    plt.ylabel('y')
    plt.savefig(f'decay_2D_concentration_field_{i}.png') # decay_ or not
    plt.close(fig)