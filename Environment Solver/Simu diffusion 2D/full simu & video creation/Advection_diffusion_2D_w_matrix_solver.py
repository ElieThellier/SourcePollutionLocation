import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

# Initialize model domains
Lx = 4     # Default 4     (Length of x domain)
Ly = 4     # Default 4     (Length of y domain)
Lt = 2     # Default 2     (Length of t domain)
Nx = 36    # Default 100   (Gridpoints in x)
Ny = 36    # Default 100   (Gridpoints in y)
Nt = 20000 # Default 20000 (Computational timesteps)
Ntp = 2000 # Default 2000  (Saved timesteps)

# Initialize model step sizes and grid
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = Lt / (Nt - 1)
x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)
t = np.linspace(0, Lt, Nt)
timesteps = np.linspace(0, Lt, Ntp)
[X, Y] = np.meshgrid(x, y)

# Initialize PDE parameters
D = 0.4                                 # Default 0.2 (Diffusion rate)
V_x = lambda t: 1.0                     # Default 2.0 (Velocity x)           
V_y = lambda t: 5.0                     # Default 3.0 (Velocity y)

# Initialize solution
u_new = np.zeros((Nx, Ny))
u_old = np.zeros((Nx, Ny))

# Source function
f = np.zeros((Nx, Ny))
lambda_1 = -1.0    # Default -1.4 (Source location x)
lambda_2 = -1.0    # Default -1.0 (Source location y)
lambda_4 =  1.0    # Default  1.0 (Source strength)
S = 25.0           # Default  25.0  (Source spread)
for i in range(1, Nx-1):
    for j in range(1, Ny-1):
        f[i, j] = lambda_4 * np.exp(-S * (((x[i] - lambda_1))**2 + ((y[j] - lambda_2))**2))

# Flatten solution
u_vec = np.zeros((Nx * Ny, Ntp))
u_new = u_new.flatten()
u_old = u_old.flatten()
f_vec = f.flatten()

# Sparse E
def sparseE(Nx, Ny):
    total_nodes = Nx * Ny
    diagonals = np.zeros((5, total_nodes))
    
    main_diag = np.ones(total_nodes)
    diagonals[2, :] = main_diag
    
    upper_diag = np.ones(total_nodes - Nx)
    diagonals[3, :-Nx] = upper_diag
    lower_diag = np.ones(total_nodes - Nx)
    diagonals[1, Nx:] = lower_diag
    
    left_diag = np.ones(total_nodes - 1)
    left_diag[np.arange(1, total_nodes) % Nx == 0] = 0
    diagonals[0, 1:] = left_diag
    right_diag = np.ones(total_nodes - 1)
    right_diag[np.arange(total_nodes - 1) % Nx == Nx - 1] = 0
    diagonals[4, :-1] = right_diag

    offsets = [-Nx, -1, 0, 1, Nx]
    return spdiags(diagonals, offsets, total_nodes, total_nodes, format='csr')

# RK4
def C_Derivative(t, u_vec, dx, dy, D, V_x, V_y, f_vec, Nx, Ny, E, e):
    alpha_y = D / (dy**2) - V_y(t) / (2 * dy)
    alpha_x = D / (dx**2) - V_x(t) / (2 * dx)
    beta = - (2 * D / (dx**2) + 2 * D / (dy**2))
    gamma_x = D / (dx**2) + V_x(t) / (2 * dx)
    gamma_y = D / (dy**2) + V_y(t) / (2 * dy)

    row = [gamma_y * e, gamma_x * e, beta * e, alpha_x * e, alpha_y * e]
    diags = [-Nx, -1, 0, 1, Nx]

    A = spdiags(row, diags, Nx * Ny, Nx * Ny, format='csr')

    k = A.dot(u_vec) + f_vec
    return k

# Iterate w/ RK4
E = sparseE(Nx, Ny)
e = np.ones(Nx * Ny)

snapstep = Nt // Ntp
snap = 0
for n in range(Nt-1):    
    k1 = C_Derivative(t[n], u_old, dx, dy, D, V_x, V_y, f_vec, Nx, Ny, E, e)
    k2 = C_Derivative(t[n] + dt / 2, u_old + k1 * dt / 2, dx, dy, D, V_x, V_y, f_vec, Nx, Ny, E, e)
    k3 = C_Derivative(t[n] + dt / 2, u_old + k2 * dt / 2, dx, dy, D, V_x, V_y, f_vec, Nx, Ny, E, e)
    k4 = C_Derivative(t[n] + dt, u_old + k3 * dt, dx, dy, D, V_x, V_y, f_vec, Nx, Ny, E, e)
    u_new = u_old + (1 / 6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    u_old = u_new
    
    if n % snapstep == 0:
        u_vec[:, snap] = u_new.flatten()
        snap += 1
    if n % 100 == 0:
        print(f"Finished with iteration: {n}")

# Unflatten solution
u = u_vec.reshape(Nx, Ny, Ntp)

""" # Save u in a file with all parameters
filename = f'c_{Nx}x{Ny}x{Nt}x{Lx}x{Ly}x{Lt}x{D}x{V_x(0)}x{V_y(0)}x{lambda_1}x{lambda_2}x{lambda_4}x{S}.npy'
np.save(filename, u)

# Example filename: c_36x36x20000x4x4x2x0.4x1.0x5.0x-1.0x-1.0x1.0x25.0.npy
print(f"Saved solution to {filename}")
 """

# Plot the solution and change y axis to match the other plot
plt.imshow(u[:, :, -1], extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], origin='lower', cmap = 'viridis')
plt.colorbar()
plt.show()
