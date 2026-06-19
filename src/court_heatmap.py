import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("../output/hip_midpoints.csv")
df = df[df["detected"] == True]

x = df["x_norm"]
y_px = df["y_norm"]
y = 1 - y_px

# Create figure
plt.figure(figsize=(6, 10))

# --- Draw badminton court (BWF proportions) ---
COURT_LENGTH = 13.40
COURT_WIDTH = 6.10
HALF_LENGTH = COURT_LENGTH / 2

singles_left = 0.46 / COURT_WIDTH
singles_right = 1 - singles_left
short_service_top = (HALF_LENGTH - 1.98) / COURT_LENGTH
short_service_bottom = (HALF_LENGTH + 1.98) / COURT_LENGTH
long_service_top = 0.76 / COURT_LENGTH
long_service_bottom = 1 - long_service_top

# Outer court boundary (doubles)
plt.plot([0, 1], [0, 0], "k-", linewidth=2)
plt.plot([0, 1], [1, 1], "k-", linewidth=2)
plt.plot([0, 0], [0, 1], "k-", linewidth=2)
plt.plot([1, 1], [0, 1], "k-", linewidth=2)

# Singles sidelines
plt.plot([singles_left, singles_left], [0, 1], "k-", linewidth=1.5)
plt.plot([singles_right, singles_right], [0, 1], "k-", linewidth=1.5)

# Short service lines (front service line, both sides of net)
plt.plot([0, 1], [short_service_top, short_service_top], color="gray", linestyle="--", linewidth=1.5)
plt.plot([0, 1], [short_service_bottom, short_service_bottom], color="gray", linestyle="--", linewidth=1.5)

# Long service lines (doubles back court)
plt.plot([0, 1], [long_service_top, long_service_top], color="gray", linestyle="--", linewidth=1)
plt.plot([0, 1], [long_service_bottom, long_service_bottom], color="gray", linestyle="--", linewidth=1)

# Center line (short service line to back boundary on each half)
plt.plot([0.5, 0.5], [0, short_service_top], "k-", linewidth=1)
plt.plot([0.5, 0.5], [short_service_bottom, 1], "k-", linewidth=1)

# Net
plt.plot([0, 1], [0.5, 0.5], "r--", linewidth=2.5)
plt.text(0.52, 0.5, "NET", color="red", fontsize=9, fontweight="bold", va="center")

# --- Heatmap overlay ---
plt.hist2d(x, y, bins=40, cmap="hot", alpha=0.6)

plt.colorbar(label="Time spent")

plt.title("Badminton Court Movement Heatmap")
plt.xlabel("Court Width")
plt.ylabel("Court Length")

plt.savefig("../output/court_heatmap.png", dpi=300)
plt.show()
