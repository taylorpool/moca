import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot3d(filename):
    sns.set_theme(context="talk", style="whitegrid")

    # Load data from data.npy
    data = np.load("data.npy")
    data = data[np.linalg.norm(data, axis=1) < 10]

    # Extract x, y, and z coordinates from the data
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Create a 3D plot
    fig = plt.figure(layout="constrained", figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z)

    # Set labels for the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)

    plt.savefig(filename)
