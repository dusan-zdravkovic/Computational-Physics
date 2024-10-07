"""
Brownian Motion and Diffusion-limited Aggregation

Author: Dusan Zdravkovic
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Initialization
Lp = 101  # Domain
Nt = 5000  # Number of time steps
t = np.arange(0, 5001, 1)  # Time array

centre_point = (Lp - 1) // 2  # Middle point of domain
xp, yp = centre_point, centre_point

x, y = [xp], [yp]


# Next move function
def nextmove(x, y):
    direction = np.random.choice([0, 1, 2, 3])  # 0=up, 1=down, 2=left, 3=right
    if direction == 0:
        y += 1
    elif direction == 1:
        y -= 1
    elif direction == 2:
        x += 1
    elif direction == 3:
        x -= 1
    return x, y


# Random walk simulation
for _ in range(Nt):
    xp_new, yp_new = nextmove(xp, yp)
    while not (0 <= xp_new < Lp and 0 <= yp_new < Lp):
        xp_new, yp_new = nextmove(xp, yp)
    x.append(xp_new)
    y.append(yp_new)
    xp, yp = xp_new, yp_new

# Plotting
plt.figure(dpi=200)
plt.title("Brownian Motion: $x$ Position Over Time")
plt.xlabel("Time (steps)")
plt.ylabel("$x$ Position")
plt.plot(t, x, color="blue")
plt.grid(True)
plt.savefig("brownian_motion_x_vs_t.png")

plt.figure(dpi=200)
plt.title("Brownian Motion: $y$ Position Over Time")
plt.xlabel("Time (steps)")
plt.ylabel("$y$ Position")
plt.plot(t, y, color="green")
plt.grid(True)
plt.savefig("brownian_motion_y_vs_t.png")

plt.figure(dpi=200, figsize=(5, 5))
plt.title("Brownian Motion Path")
plt.xlabel("$x$ Position")
plt.ylabel("$y$ Position")
plt.plot(x, y, color="purple", marker="o", linestyle="-", markersize=2)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True)
plt.savefig("brownian_motion_x_vs_y.png")

# Diffusion-limited aggregation
anchored = np.zeros((Lp, Lp), dtype=int)
anchored_points = [[], []]

while anchored[centre_point, centre_point] != 1:
    xp, yp = centre_point, centre_point
    while (
        0 < xp < Lp - 1
        and 0 < yp < Lp - 1
        and not (
            anchored[xp - 1, yp]
            or anchored[xp + 1, yp]
            or anchored[xp, yp - 1]
            or anchored[xp, yp + 1]
        )
    ):
        xp, yp = nextmove(xp, yp)

    anchored_points[0].append(xp)
    anchored_points[1].append(yp)
    anchored[xp, yp] = 1

plt.figure(dpi=200)
plt.imshow(anchored, cmap="viridis")
plt.plot(50, 50, marker="o", color="red", markersize=8)
plt.title("Diffusion-limited Aggregation Visualization")
plt.xlabel("x Position")
plt.ylabel("y Position")
plt.grid(True)
plt.savefig("diffusion_limited_aggregation.png")
plt.show()
