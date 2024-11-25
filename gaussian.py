import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
length = 100  
nodes = 50 
a = 1 
time = 100  

# Initialization 
dx = length / (nodes - 1)
dy = length / (nodes - 1)

# Initial condition
A = 200  
x0, y0 = length / 2, length / 2 
sigma_x, sigma_y = 4, 4 

# Inisialisasi grid dan suhu awal 
x = np.linspace(0, length, nodes)
y = np.linspace(0, length, nodes)
X, Y = np.meshgrid(x, y)
T = A * np.exp(-((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2))) + 10


# Boundary condition
T[0, :] = 20  
T[-1, :] = 20  
T[:, 0] = 20 
T[:, -1] = 20 

# Stability check function
def check_stability(dx, dy, a):
    max_dt = min(dx**2, dy**2) / (4 * a)
    return max_dt

# Calculate time step
dt = check_stability(dx, dy, a)
if dt:
    print('Kondisi stabilitas terpenuhi')
else:
    print('Kondisi stabilitas tidak terpenuhi')

def temperature(T, a, dt, dx, dy):
    T_new = np.copy(T)
    T_new[1:-1, 1:-1] += a * dt * (
        (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 +
        (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2)
    return T_new

# Set up the plot
fig, axis = plt.subplots()
pcm = axis.pcolormesh(X, Y, T, cmap=plt.cm.jet, vmin=0, vmax=100)
plt.colorbar(pcm, ax=axis)

# Animation function
def animate(frame):
    global T
    T = temperature(T, a, dt, dx, dy)
    pcm.set_array(T.ravel())  
    axis.set_title(f"Temperature Distribution at t: {frame * dt:.3f} s")

    average_temp = np.mean(T)
    print(f"t: {frame * dt:.3f} [s], Average temperature: {average_temp:.2f} Celsius")
    return pcm,

# Create the animation
anim = FuncAnimation(fig, animate, frames=int(time / dt), interval=50, blit=False, repeat=False)

plt.show()
