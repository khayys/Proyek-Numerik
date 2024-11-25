import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Defining our problem
a = 1 # Difusivitas termal
length = 100  # mm
time = 100  # seconds
nodes = 50

# Initialization 
dx = length / (nodes - 1)
dy = length / (nodes - 1)

# Fungsi untuk memeriksa stabilitas
def check_stability(dx, dy, a):
    # Menggunakan syarat stabilitas untuk heat diffusion 2D
    max_dt = min(dx**2, dy**2) / (4 * a)
    return max_dt

# Menghitung dt berdasarkan kondisi stabilitas
dt = check_stability(dx, dy, a)

t_nodes = int(time / dt) + 1  # jumlah langkah waktu
u = np.zeros((nodes, nodes))

# Initial condition for the plate
u[:, :] = 20

# Function to perform one time step in the simulation
def update_temperature(T, a, dt, dx, dy):
    # Salin grid saat ini untuk pembaruan
    T_new = np.copy(T)
    # Update suhu di grid interior (menggunakan slicing untuk efisiensi)
    T_new[1:-1, 1:-1] += a * dt * (
        (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 +
        (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2)

    # Heat Source di Tengah
    center_x, center_y = T.shape[0] // 2, T.shape[1] // 2
    T_new[center_x-1:center_x+2, center_y-1:center_y+2] = 90
    
    # Boundary conditions
    T_new[0, :] = 30
    T_new[-1, :] = 30
    T_new[:, 0] = 30
    T_new[:, -1] = 30

    return T_new

# Setup for the plot
fig, axis = plt.subplots()
pcm = axis.pcolormesh(u, cmap=plt.cm.jet, vmin=0, vmax=100)
plt.colorbar(pcm, ax=axis)

# Update function for FuncAnimation
def animate(frame):
    global u
    u = update_temperature(u, a, dt, dx, dy)  
    pcm.set_array(u.ravel())  
    axis.set_title(f"Temperature Distribution at t: {frame * dt:.3f} s")

    # Calculate and print the average temperature 
    average_temp = np.mean(u)
    print(f"t: {frame * dt:.3f} [s], Average temperature: {average_temp:.2f} Celsius")
    
    return pcm,  

# Create the animation
anim = FuncAnimation(fig, animate, frames=int(time / dt), interval=50, repeat=False)

plt.show()
