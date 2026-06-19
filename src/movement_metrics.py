from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROLLING_WINDOW = 30
HIGHLIGHT_HALF_WIDTH = 15
TOP_PERIODS = 5

ZONE_TOP = "Top of frame"
ZONE_MIDDLE = "Middle of frame"
ZONE_BOTTOM = "Bottom of frame"
ZONE_NOTE = "Note: Zones are in image coordinate space, not real court geometry."
VERTICAL_ZONE_ORDER = [ZONE_TOP, ZONE_MIDDLE, ZONE_BOTTOM]
HORIZONTAL_ZONE_ORDER = ["Left", "Right"]

RALLY_FPS = 30
RALLY_WINDOW_SECONDS = 3
RALLY_WINDOW_FRAMES = RALLY_FPS * RALLY_WINDOW_SECONDS


def load_tracking_positions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.dropna(subset=["x_px", "y_px"]).sort_values("frame").reset_index(drop=True)


def build_movement_series(csv_path: Path) -> pd.DataFrame:
    df = load_tracking_positions(csv_path)

    x = df["x_px"].astype(float).to_numpy()
    y = df["y_px"].astype(float).to_numpy()
    frames = df["frame"].astype(int).to_numpy()

    if len(frames) < 2:
        return pd.DataFrame(columns=["frame", "movement"])

    movement = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return pd.DataFrame({"frame": frames[1:], "movement": movement})


def calculate_movement_metrics(csv_path: Path) -> dict:
    movement_df = build_movement_series(csv_path)
    frames_analyzed = len(movement_df) + 1 if not movement_df.empty else 0

    if movement_df.empty:
        return {
            "frames_analyzed": frames_analyzed,
            "total_distance": 0.0,
            "average_movement": 0.0,
            "max_movement": 0.0,
        }

    distances = movement_df["movement"].to_numpy()

    return {
        "frames_analyzed": frames_analyzed,
        "total_distance": float(distances.sum()),
        "average_movement": float(distances.mean()),
        "max_movement": float(distances.max()),
    }


def get_top_movement_periods(
    movement_df: pd.DataFrame,
    n: int = TOP_PERIODS,
    min_separation: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    candidates = movement_df.sort_values("smoothed", ascending=False)
    selected = []

    for _, row in candidates.iterrows():
        if all(abs(row["frame"] - period["frame"]) >= min_separation for period in selected):
            selected.append(row)
        if len(selected) == n:
            break

    return pd.DataFrame(selected)


def plot_movement_over_time(
    csv_path: Path,
    output_path: Path,
    rolling_window: int = ROLLING_WINDOW,
) -> Path:
    movement_df = build_movement_series(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if movement_df.empty:
        raise ValueError("Not enough movement data to plot.")

    movement_df["smoothed"] = (
        movement_df["movement"]
        .rolling(window=rolling_window, min_periods=1, center=True)
        .mean()
    )

    top_periods = get_top_movement_periods(movement_df)
    highlight_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        movement_df["frame"],
        movement_df["movement"],
        color="#bdc3c7",
        linewidth=1,
        label="Frame movement",
    )
    ax.plot(
        movement_df["frame"],
        movement_df["smoothed"],
        color="#2c3e50",
        linewidth=2,
        label=f"{rolling_window}-frame rolling average",
    )

    for _, row in top_periods.iterrows():
        color = highlight_colors[len(ax.patches) % len(highlight_colors)]
        peak_frame = int(row["frame"])
        rank = len(ax.patches) + 1

        ax.axvspan(
            peak_frame - HIGHLIGHT_HALF_WIDTH,
            peak_frame + HIGHLIGHT_HALF_WIDTH,
            color=color,
            alpha=0.2,
            label=f"Top period {rank} (frame {peak_frame})",
        )
        ax.scatter(
            peak_frame,
            row["smoothed"],
            color=color,
            s=60,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_title("Movement Over Time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Movement (pixels)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def build_center_distance_series(csv_path: Path) -> tuple[pd.DataFrame, float, float]:
    df = load_tracking_positions(csv_path)

    if df.empty:
        return pd.DataFrame(columns=["frame", "distance_from_center"]), 0.0, 0.0

    center_x = float(df["x_px"].mean())
    center_y = float(df["y_px"].mean())

    distances = np.sqrt(
        (df["x_px"].astype(float) - center_x) ** 2
        + (df["y_px"].astype(float) - center_y) ** 2
    )

    series = pd.DataFrame(
        {
            "frame": df["frame"].astype(int),
            "distance_from_center": distances,
        }
    )

    return series, center_x, center_y


def calculate_center_recovery_metrics(csv_path: Path) -> dict:
    series, center_x, center_y = build_center_distance_series(csv_path)

    if series.empty:
        return {
            "center_x": center_x,
            "center_y": center_y,
            "average_distance": 0.0,
            "maximum_distance": 0.0,
            "std_deviation": 0.0,
        }

    distances = series["distance_from_center"].to_numpy()

    return {
        "center_x": center_x,
        "center_y": center_y,
        "average_distance": float(distances.mean()),
        "maximum_distance": float(distances.max()),
        "std_deviation": float(distances.std(ddof=0)),
    }


def plot_center_distance_over_time(
    csv_path: Path,
    output_path: Path,
    rolling_window: int = ROLLING_WINDOW,
) -> Path:
    series, center_x, center_y = build_center_distance_series(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if series.empty:
        raise ValueError("Not enough position data to plot center distance.")

    series["smoothed"] = (
        series["distance_from_center"]
        .rolling(window=rolling_window, min_periods=1, center=True)
        .mean()
    )

    average_distance = series["distance_from_center"].mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        series["frame"],
        series["distance_from_center"],
        color="#bdc3c7",
        linewidth=1,
        label="Distance from center",
    )
    ax.plot(
        series["frame"],
        series["smoothed"],
        color="#8e44ad",
        linewidth=2,
        label=f"{rolling_window}-frame rolling average",
    )
    ax.axhline(
        average_distance,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        label=f"Average distance ({average_distance:.1f} px)",
    )

    ax.set_title("Center Distance Over Time")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Distance From Center (pixels)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def print_movement_metrics(metrics: dict) -> None:
    print(f"Frames analyzed: {metrics['frames_analyzed']}")
    print(f"Total distance traveled: {metrics['total_distance']:.2f} pixels")
    print(f"Average movement per frame: {metrics['average_movement']:.2f} pixels")
    print(f"Maximum movement in a frame: {metrics['max_movement']:.2f} pixels")


def print_center_recovery_metrics(metrics: dict) -> None:
    print("\nCenter Recovery Analysis\n")
    print("Center Position:")
    print(f"({metrics['center_x']:.1f}, {metrics['center_y']:.1f})\n")
    print("Average Distance From Center:")
    print(f"{metrics['average_distance']:.1f} px\n")
    print("Maximum Distance From Center:")
    print(f"{metrics['maximum_distance']:.1f} px\n")
    print("Standard Deviation:")
    print(f"{metrics['std_deviation']:.1f} px")


def compute_zone_occupancy_metrics(csv_path: Path) -> dict:
    df = load_tracking_positions(csv_path)

    if df.empty:
        raise ValueError("Not enough position data for zone occupancy analysis.")

    x_min = float(df["x_px"].min())
    x_max = float(df["x_px"].max())
    y_min = float(df["y_px"].min())
    y_max = float(df["y_px"].max())

    image_width = x_max - x_min
    image_height = y_max - y_min

    if image_width == 0:
        image_width = 1.0
    if image_height == 0:
        image_height = 1.0

    top_boundary = y_min + image_height / 3
    bottom_boundary = y_min + 2 * image_height / 3
    center_x = x_min + image_width / 2

    vertical_zone = np.select(
        [df["y_px"] < top_boundary, df["y_px"] < bottom_boundary],
        [ZONE_TOP, ZONE_MIDDLE],
        default=ZONE_BOTTOM,
    )
    horizontal_zone = np.where(df["x_px"] < center_x, "Left", "Right")

    total_frames = len(df)
    vertical_counts = pd.Series(vertical_zone).value_counts()
    horizontal_counts = pd.Series(horizontal_zone).value_counts()

    vertical_pct = {
        zone: (vertical_counts.get(zone, 0) / total_frames) * 100
        for zone in VERTICAL_ZONE_ORDER
    }
    horizontal_pct = {
        zone: (horizontal_counts.get(zone, 0) / total_frames) * 100
        for zone in HORIZONTAL_ZONE_ORDER
    }

    return {
        "top_pct": vertical_pct[ZONE_TOP],
        "middle_pct": vertical_pct[ZONE_MIDDLE],
        "bottom_pct": vertical_pct[ZONE_BOTTOM],
        "left_pct": horizontal_pct["Left"],
        "right_pct": horizontal_pct["Right"],
        "image_width": image_width,
        "image_height": image_height,
        "frames_analyzed": total_frames,
    }


def zone_occupancy_analysis(csv_path: Path, output_path: Path) -> dict:
    metrics = compute_zone_occupancy_metrics(csv_path)

    print("\nImage-Based Zone Occupancy:")
    print(f"{ZONE_TOP}: {metrics['top_pct']:.1f}%")
    print(f"{ZONE_MIDDLE}: {metrics['middle_pct']:.1f}%")
    print(f"{ZONE_BOTTOM}: {metrics['bottom_pct']:.1f}%")
    print()
    print(f"Left: {metrics['left_pct']:.1f}%")
    print(f"Right: {metrics['right_pct']:.1f}%")
    print()
    print(ZONE_NOTE)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(
        VERTICAL_ZONE_ORDER,
        [metrics["top_pct"], metrics["middle_pct"], metrics["bottom_pct"]],
        color=["#3498db", "#2ecc71", "#e74c3c"],
    )
    axes[0].set_title("Image-Based Zones (Vertical)")
    axes[0].set_ylabel("Time Spent (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        HORIZONTAL_ZONE_ORDER,
        [metrics["left_pct"], metrics["right_pct"]],
        color=["#9b59b6", "#f39c12"],
    )
    axes[1].set_title("Image-Based Zones (Horizontal)")
    axes[1].set_ylabel("Time Spent (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        for bar in ax.patches:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle("Image-Based Zone Occupancy", fontsize=14)
    fig.text(0.5, 0.01, ZONE_NOTE, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return metrics


def compute_rally_intensity(
    csv_path: Path,
    output_csv: Path,
    output_png: Path,
    fps: int = RALLY_FPS,
    window_seconds: int = RALLY_WINDOW_SECONDS,
    smooth_window: int = ROLLING_WINDOW,
) -> list[tuple[int, float]]:
    movement_df = build_movement_series(csv_path)
    window_frames = fps * window_seconds

    if movement_df.empty:
        raise ValueError("Not enough movement data for rally intensity analysis.")

    max_frame = int(movement_df["frame"].max())
    rally_windows = []

    for window_start in range(0, max_frame + 1, window_frames):
        window_end = window_start + window_frames
        window_data = movement_df[
            (movement_df["frame"] >= window_start) & (movement_df["frame"] < window_end)
        ]

        total_movement = float(window_data["movement"].sum())
        avg_movement = (
            float(window_data["movement"].mean()) if not window_data.empty else 0.0
        )
        intensity_score = total_movement

        rally_windows.append(
            {
                "window_start_frame": window_start,
                "intensity_score": intensity_score,
                "total_movement": total_movement,
                "average_movement_per_frame": avg_movement,
            }
        )

    rally_df = pd.DataFrame(rally_windows)
    results = list(zip(rally_df["window_start_frame"], rally_df["intensity_score"]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rally_df.to_csv(output_csv, index=False)

    rally_df["smoothed_intensity"] = (
        rally_df["intensity_score"]
        .rolling(window=smooth_window, min_periods=1, center=True)
        .mean()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        rally_df["window_start_frame"],
        rally_df["intensity_score"],
        color="#bdc3c7",
        linewidth=1,
        label="Intensity score",
    )
    ax.plot(
        rally_df["window_start_frame"],
        rally_df["smoothed_intensity"],
        color="#c0392b",
        linewidth=2,
        label=f"{smooth_window}-window rolling average",
    )
    ax.set_title("Rally Intensity Over Time")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Intensity Score (total movement)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    csv_file = project_root / "output" / "hip_midpoints.csv"
    movement_chart = project_root / "output" / "movement_over_time.png"
    center_chart = project_root / "output" / "center_distance_over_time.png"
    zone_chart = project_root / "output" / "zone_occupancy.png"
    rally_csv = project_root / "output" / "rally_intensity.csv"
    rally_chart = project_root / "output" / "rally_intensity.png"

    results = calculate_movement_metrics(csv_file)
    print_movement_metrics(results)

    saved_movement_chart = plot_movement_over_time(csv_file, movement_chart)
    print(f"Saved chart to {saved_movement_chart}")

    center_results = calculate_center_recovery_metrics(csv_file)
    print_center_recovery_metrics(center_results)

    saved_center_chart = plot_center_distance_over_time(csv_file, center_chart)
    print(f"\nSaved chart to {saved_center_chart}")

    zone_occupancy_analysis(csv_file, zone_chart)
    print(f"Saved chart to {zone_chart}")

    compute_rally_intensity(csv_file, rally_csv, rally_chart)
    print(f"Saved rally intensity to {rally_csv}")
    print(f"Saved chart to {rally_chart}")
