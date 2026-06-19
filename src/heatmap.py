import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("../output/hip_midpoints.csv")
df = df[df["detected"] == True]

x = df["x_px"]
y = df["y_px"]

# Create heatmap
plt.figure(figsize=(6, 10))
plt.hist2d(x, y, bins=50, cmap="hot")

plt.colorbar(label="Time spent")

plt.title("Badminton Player Movement Heatmap")
plt.xlabel("Court X Position")
plt.ylabel("Court Y Position")

plt.savefig("../output/heatmap.png")
plt.show()
