import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameter
a = 10 
length = 100 
time = 100  
nodes = 50

# Initialization 
dx = length / (nodes - 1)
dy = length / (nodes - 1)

# Check stabilitas
def check_stability(dx, dy, a):
    max_dt = min(dx**2, dy**2) / (4 * a)
    return max_dt

# Hitung dt
dt = check_stability(dx, dy, a)
if dt:
    print('Kondisi stabilitas terpenuhi')
else:
    print('Kondisi stabilitas tidak terpenuhi')

# Inisialisasi grid
t_nodes = int(time / dt) + 1  
u = np.zeros((nodes, nodes))

# Initial condition for the plate
u[:, :] = 20

def update_temperature(T, a, dt, dx, dy):
    T_new = np.copy(T)
    T_new[1:-1, 1:-1] += a * dt * (
        (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 +
        (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2)

    # Heat Source di tengah
    center_x, center_y = T.shape[0] // 2, T.shape[1] // 2
    T_new[center_x-1:center_x+2, center_y-1:center_y+2] = 90

    return T_new

# Setup for the plot
fig, axis = plt.subplots()
pcm = axis.pcolormesh(u, cmap=plt.cm.jet, vmin=0, vmax=100)
plt.colorbar(pcm, ax=axis)

# Animation Function
def animate(frame):
    global u
    u = update_temperature(u, a, dt, dx, dy)  
    pcm.set_array(u.ravel())  
    axis.set_title(f"Temperature Distribution at t: {frame * dt:.3f} s")

    average_temp = np.mean(u)
    print(f"t: {frame * dt:.3f} [s], Average temperature: {average_temp:.2f} Celsius")
    return pcm,  

# Create the animation
anim = FuncAnimation(fig, animate, frames=int(time / dt), interval=50, repeat=False)

plt.show()

# Save the animation as a gif
anim.save('animation.gif', writer='pillow', fps=50)