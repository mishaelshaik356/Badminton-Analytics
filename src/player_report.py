from pathlib import Path

import matplotlib.pyplot as plt

from movement_metrics import (
    ROLLING_WINDOW,
    VERTICAL_ZONE_ORDER,
    ZONE_BOTTOM,
    ZONE_MIDDLE,
    ZONE_NOTE,
    ZONE_TOP,
    build_movement_series,
    calculate_center_recovery_metrics,
    calculate_movement_metrics,
    compute_zone_occupancy_metrics,
)


def build_player_report(
    movement: dict,
    center: dict,
    zones: dict,
) -> str:
    lines = [
        "========================",
        "PLAYER MATCH REPORT",
        "========================",
        "",
        "Movement:",
        f"- Total Distance: {movement['total_distance']:.2f} px",
        f"- Avg Movement per Frame: {movement['average_movement']:.2f} px",
        "",
        "Positioning (Image-Based Zones):",
        f"- {ZONE_TOP}: {zones['top_pct']:.1f}%",
        f"- {ZONE_MIDDLE}: {zones['middle_pct']:.1f}%",
        f"- {ZONE_BOTTOM}: {zones['bottom_pct']:.1f}%",
        f"- Left: {zones['left_pct']:.1f}%",
        f"- Right: {zones['right_pct']:.1f}%",
        "",
        "Stability:",
        f"- Avg Distance from Center: {center['average_distance']:.1f} px",
        f"- Max Distance: {center['maximum_distance']:.1f} px",
        "",
        ZONE_NOTE,
        "",
    ]
    return "\n".join(lines)


def save_player_summary_dashboard(
    csv_path: Path,
    zones: dict,
    output_path: Path,
    rolling_window: int = ROLLING_WINDOW,
) -> Path:
    movement_df = build_movement_series(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if movement_df.empty:
        raise ValueError("Not enough movement data to create summary dashboard.")

    movement_df["smoothed"] = (
        movement_df["movement"]
        .rolling(window=rolling_window, min_periods=1, center=True)
        .mean()
    )

    zone_labels = VERTICAL_ZONE_ORDER + ["Left", "Right"]
    zone_values = [
        zones["top_pct"],
        zones["middle_pct"],
        zones["bottom_pct"],
        zones["left_pct"],
        zones["right_pct"],
    ]
    zone_colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].bar(zone_labels, zone_values, color=zone_colors)
    axes[0].set_title("Image-Based Zone Occupancy")
    axes[0].set_ylabel("Time Spent (%)")
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis="x", labelrotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    for bar in axes[0].patches:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1].plot(
        movement_df["frame"],
        movement_df["smoothed"],
        color="#2c3e50",
        linewidth=2,
    )
    axes[1].set_title("Movement Over Time")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Movement (pixels)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Player Summary Dashboard", fontsize=14)
    fig.text(0.5, 0.01, ZONE_NOTE, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def generate_player_report(
    csv_path: Path,
    text_output: Path,
    image_output: Path,
) -> str:
    movement = calculate_movement_metrics(csv_path)
    center = calculate_center_recovery_metrics(csv_path)
    zones = compute_zone_occupancy_metrics(csv_path)

    report = build_player_report(movement, center, zones)

    text_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.write_text(report)
    save_player_summary_dashboard(csv_path, zones, image_output)

    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    csv_file = project_root / "output" / "hip_midpoints.csv"
    report_file = project_root / "output" / "player_report.txt"
    summary_image = project_root / "output" / "player_summary.png"

    report_text = generate_player_report(csv_file, report_file, summary_image)
    print(report_text)
    print(f"Saved report to {report_file}")
    print(f"Saved dashboard to {summary_image}")
