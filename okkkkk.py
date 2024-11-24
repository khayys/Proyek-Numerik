import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Defining our problem
a = 80  # Difusivitas termal
length = 100  # mm
time = 10  # seconds
nodes = 100

# Initialization 
dx = length / (nodes - 1)
dy = length / (nodes - 1)

# Fungsi untuk memeriksa stabilitas
def check_stability(dx, dy, a):
    # Menggunakan syarat stabilitas untuk heat diffusion 2D
    max_dt = (dx**2 * dy**2) / (2 * a * (dx**2 + dy**2))
    return max_dt

# Menghitung dt berdasarkan kondisi stabilitas
dt = check_stability(dx, dy, a)

t_nodes = int(time / dt) + 1  # jumlah langkah waktu
u = np.zeros((nodes, nodes))

# Initial condition for the plate
u[:, :] = 20  # Suhu awal merata di seluruh grid

# Fungsi untuk menambahkan distribusi panas secara alami dari pusat
def heat_source_distribution(T, center_x, center_y, strength, time_step):
    # Fungsi distribusi panas yang menyebar lebih realistis
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            # Menghitung jarak dari pusat
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if distance <= 3:  # Radius pengaruh panas
                # Panas ditambahkan berdasarkan jarak dari pusat dan waktu
                T[i, j] += strength * np.exp(-distance**2 / (2 * 5**2)) * np.exp(-time_step/2)

# Function to perform one time step in the simulation
def update_temperature(T, a, dt, dx, dy, T_top, T_bottom, T_left, T_right, time_step):
    # Salin grid saat ini untuk pembaruan
    T_new = np.copy(T)
    
    # Update suhu di grid interior (menggunakan slicing untuk efisiensi)
    T_new[1:-1, 1:-1] += a * dt * (
        (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 +
        (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2)

    # Distribusi panas di pusat dengan waktu
    center_x, center_y = T.shape[0] // 2, T.shape[1] // 2
    heat_source_strength = 100  # Kekuatan sumber panas
    heat_source_distribution(T_new, center_x, center_y, heat_source_strength, time_step)

    # Boundary conditions
    T_new[0, :] = T_top  # Atas
    T_new[-1, :] = T_bottom  # Bawah
    T_new[:, 0] = T_left  # Kiri
    T_new[:, -1] = T_right  # Kanan

    return T_new

# Setup for the plot
fig, axis = plt.subplots()
pcm = axis.pcolormesh(u, cmap=plt.cm.jet, vmin=0, vmax=200)
plt.colorbar(pcm, ax=axis)

# Update function for FuncAnimation
def animate(frame):
    global u
    u = update_temperature(u, a, dt, dx, dy, T_top=20, T_bottom=20, T_left=20, T_right=20, time_step=frame)
    pcm.set_array(u.ravel())  # Update plot dengan array suhu baru
    axis.set_title(f"Temperature Distribution at t: {frame * dt:.3f} s")

    # Calculate and print the average temperature for the current time step
    average_temp = np.mean(u)
    print(f"t: {frame * dt:.3f} [s], Average temperature: {average_temp:.2f} Celsius")
    
    return pcm,  # Return a tuple of updated artist(s)

# Create the animation
anim = FuncAnimation(fig, animate, frames=int(time / dt), interval=50, repeat=True)

plt.show()
