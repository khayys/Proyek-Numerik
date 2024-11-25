import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
Lx, Ly = 10, 10  # Panjang dalam x dan y
n = 50  # Nodes
dx, dy = Lx / (n - 1), Ly / (n - 1)
a = 100  # Konstanta difusi
time = 4  # Waktu total simulasi

# Initial condition
A = 100  # Amplitudo
x0, y0 = Lx / 2, Ly / 2  # Pusat distribusi
sigma_x, sigma_y = 1, 1  # Lebar distribusi di sumbu x dan y

# Inisialisasi grid dan suhu awal (distribusi Gaussian)
x = np.linspace(0, Lx, n)
y = np.linspace(0, Ly, n)
X, Y = np.meshgrid(x, y)
T = A * np.exp(-((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2))) + 10

# np.linspace(0, 100, n)

# Boundary condition
T[0, :] = 20  # batas atas
T[-1, :] = 20  # batas bawah
T[:, 0] = 20  # batas kiri
T[:, -1] = 20 # batas kanan

# Stability check function
def check_stability(dx, dy, a):
    max_dt = (dx**2 * dy**2) / (2 * a * (dx**2 + dy**2))
    return max_dt

# Calculate time step
dt = check_stability(dx, dy, a)
if dt:
    print('Kondisi stabilitas terpenuhi')
else:
    print('Kondisi stabilitas tidak terpenuhi')

# Update temperature function (finite difference method)
def update_temperature(T, a, dt, dx, dy):
    T_new = np.copy(T)
    T_new[1:-1, 1:-1] += a * dt * (
        (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 +
        (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
    )
    return T_new

# Set up the plot
fig, axis = plt.subplots()
pcm = axis.pcolormesh(X, Y, T, cmap=plt.cm.jet, vmin=0, vmax=100)
plt.colorbar(pcm, ax=axis)

# Animation function
def animate(frame):
    global T
    T = update_temperature(T, a, dt, dx, dy)
    pcm.set_array(T.ravel())  # Update the pcolormesh plot data
    axis.set_title(f"Temperature Distribution at t: {frame * dt:.3f} s")

    # Calculate and print the average temperature for the current time step
    average_temp = np.mean(T)
    print(f"t: {frame * dt:.3f} [s], Average temperature: {average_temp:.2f} Celsius")
    return pcm,

# Create the animation
anim = FuncAnimation(fig, animate, frames=int(time / dt), interval=50, blit=True, repeat=True)

plt.show()
