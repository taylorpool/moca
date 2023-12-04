import numpy as np
import matplotlib.pyplot as plt

# Load data from data.npy
data = np.load("data.npy")
data = data[np.linalg.norm(data, axis=1) < 10]
print(data.shape)

# Extract x, y, and z coordinates from the data
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z)

# Set labels for the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Show the plot
plt.show()
