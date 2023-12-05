import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid")

# Load data from data.npy
data = np.load("data.npy")
# data = data[np.linalg.norm(data, axis=1) < 10]

# Extract x, y, and z coordinates from the data
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Create a 3D plot
fig = plt.figure(layout="constrained", figsize=(12, 6))
ax = fig.add_subplot(122, projection="3d")
ax.scatter(x, y, z)

# Set labels for the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax = fig.add_subplot(121)
data2 = [
    32.55422541053893,
    7.55341892922651,
    5.923936142674917,
    5.799755251539001,
    5.730829565089122,
    5.668672380164414,
    5.615528663891578,
    5.569985295001045,
    5.531628506197591,
    5.499510091919993,
    5.472582662626314,
    5.449852521257917,
    5.430485454546906,
    5.413820333171663,
    5.399345560088179,
    5.386666597049043,
    5.375477048099056,
    5.365536221279892,
    5.356652619400751,
    5.348671985337364,
]
plt.plot(data2, marker=".")
plt.xlabel("Iterations")
plt.ylabel("Residual")
plt.savefig("points.png")
