import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gtsam
from src.sfm.colmap import frontend
import os


def plot_pose3_on_axes(axes, pose, axis_length=0.1, scale=1):
    """
    Plot a 3D pose on given axis `axes` with given `axis_length`.

    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        point (gtsam.Point3): The point to be plotted.
        linespec (string): String representing formatting options for Matplotlib.
        P (numpy.ndarray): Marginal covariance matrix to plot the uncertainty of the estimation.
    From: https://github.com/borglab/gtsam/blob/develop/python/gtsam/utils/plot.py
    """
    # get rotation and translation (center)
    gRp = pose.rotation().matrix()  # rotation from pose to global
    origin = pose.translation()

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], "r-")

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], "g-")

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin[np.newaxis], z_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], "b-")


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
    data = np.load(filename + ".npz")
    pts = data["arr_0"]
    poses = data["arr_1"]

    # Extract x, y, and z coordinates from the data
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # Create a 3D plot
    fig = plt.figure(layout="constrained", figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z)

    # Plot camera poses
    for p in poses:
        qx, qy, qz, qw, tx, ty, tz = p
        pose = gtsam.Pose3(gtsam.Rot3(qw, qx, qy, qz), gtsam.Point3(tx, ty, tz))
        plot_pose3_on_axes(ax, pose)

    # Set labels for the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)

    plt.savefig(filename + ".png")


def plot_open3d(filename):
    import open3d as o3d

    # Flips everything so it's easily viewable by default
    flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    # Load data from data.npy
    data = np.load(filename + ".npz")
    allpts = data["arr_0"]
    lm_id = allpts[:, 0]
    pts = allpts[:, 1:]

    poses_vec = data["arr_1"]
    poses = []

    # Load pixels from frontend
    sfm_data = frontend(filename)

    pixels = []
    for id in lm_id:
        factor = sfm_data.factors[sfm_data.lm_lookup[id][0]]
        id_pose = factor.id_pose
        file = sfm_data.filenames[id_pose]
        img = plt.imread(os.path.join(filename, file))
        pix = img[int(np.round(factor.v)), int(np.round(factor.u))] / 255
        pixels.append(pix)

    pixels = np.array(pixels)

    for p in poses_vec:
        qx, qy, qz, qw, tx, ty, tz = p
        pose = gtsam.Pose3(gtsam.Rot3(qw, qx, qy, qz), gtsam.Point3(tx, ty, tz))
        poses.append(pose.matrix())

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(pixels)
    pc.transform(flip)

    geo = [pc]

    for p in poses:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        frame.transform(np.linalg.inv(p))
        frame.transform(flip)
        geo.append(frame)

    o3d.visualization.draw_geometries(geo)


if __name__ == "__main__":
    plot_open3d("moose")
