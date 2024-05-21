import numpy as np
import matplotlib.pyplot as plt

D = 1


def brownian_motion(time, partition, ornstein=False):
    time_fraction = time / partition
    x = 0
    t_vals = np.linspace(0, time, partition)
    x_vals = []
    for t in t_vals:
        x_vals.append(x)
        x += np.sqrt(2 * D) * np.random.normal(loc=0, scale=np.sqrt(time_fraction), size=None)
        if ornstein:
            x -= x * time_fraction

    return t_vals, np.array(x_vals)


def mean_squared_displacement(time, partition, num_trajectories):
    t_vals = np.linspace(0, time, partition)
    trajectories = []
    for _ in range(num_trajectories):
        trajectory = brownian_motion(time, partition)[1]
        trajectories.append(trajectory)
    trajectories = np.array(trajectories)

    mean = np.array([sum(trajectories[:, q]) for q in range(len(trajectories))]) / len(trajectories)
    squared_displacement = (trajectories - mean) ** 2
    return t_vals, np.array([sum(squared_displacement[:, q]) for q in range(len(squared_displacement))]) / len(squared_displacement)


fig, ax = plt.subplots(dpi=500)
for _ in range(5):
    ax.plot(*brownian_motion(10, 1000))

fig.tight_layout()
fig.savefig('brownian_motion.png')

fig, ax = plt.subplots(dpi=500)
ax.plot(*mean_squared_displacement(10, 1000, 1000))

fig.tight_layout()
fig.savefig('mean_squared_displacement.png')

fig, ax = plt.subplots(dpi=500)
for _ in range(3):
    ax.plot(*brownian_motion(10, 1000, True))

fig.tight_layout()
fig.savefig('brownian_motion_ornstein.png')
