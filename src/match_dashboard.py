from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from movement_metrics import (
    RALLY_FPS,
    RALLY_WINDOW_SECONDS,
    ROLLING_WINDOW,
    VERTICAL_ZONE_ORDER,
    ZONE_NOTE,
    build_center_distance_series,
    build_movement_series,
    calculate_center_recovery_metrics,
    calculate_movement_metrics,
    compute_zone_occupancy_metrics,
    load_tracking_positions,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIP_CSV = PROJECT_ROOT / "output" / "hip_midpoints.csv"
RALLY_CSV = PROJECT_ROOT / "output" / "rally_intensity.csv"
VIDEO_PATH = PROJECT_ROOT / "videos" / "practice.mp4"
VIDEO_FALLBACK = PROJECT_ROOT / "videos" / "practice copy.mp4"
DISPLAY_MAX_WIDTH = 960
RALLY_WINDOW_FRAMES = RALLY_FPS * RALLY_WINDOW_SECONDS


def resolve_video_path() -> Path | None:
    if VIDEO_PATH.exists():
        return VIDEO_PATH
    if VIDEO_FALLBACK.exists():
        return VIDEO_FALLBACK
    return None


def build_rally_intensity_dataframe(hip_csv: Path) -> pd.DataFrame:
    movement_df = build_movement_series(hip_csv)
    window_frames = RALLY_WINDOW_FRAMES

    if movement_df.empty:
        return pd.DataFrame(
            columns=[
                "window_start_frame",
                "intensity_score",
                "total_movement",
                "average_movement_per_frame",
            ]
        )

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

        rally_windows.append(
            {
                "window_start_frame": window_start,
                "intensity_score": total_movement,
                "total_movement": total_movement,
                "average_movement_per_frame": avg_movement,
            }
        )

    return pd.DataFrame(rally_windows)


@st.cache_resource
def get_video_metadata(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "total_frames": max(total_frames, 1),
        "fps": fps,
        "width": max(width, 1),
        "height": max(height, 1),
    }


def read_video_frame(video_path: str, frame_index: int, max_width: int = DISPLAY_MAX_WIDTH) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    cap.release()

    if not success:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width = frame_rgb.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame_rgb = cv2.resize(frame_rgb, (max_width, int(height * scale)))

    return frame_rgb


def build_movement_heatmap_frame(
    positions_df: pd.DataFrame,
    frame_idx: int,
    width: int,
    height: int,
    max_width: int = DISPLAY_MAX_WIDTH,
) -> np.ndarray:
    """Build a cumulative movement heatmap using image pixel coordinates."""
    data = positions_df[positions_df["frame"] <= frame_idx]

    scale = min(1.0, max_width / width)
    display_w = max(int(width * scale), 1)
    display_h = max(int(height * scale), 1)
    canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)

    if data.empty:
        return canvas

    x = (data["x_px"].astype(float) * scale).astype(int)
    y = (data["y_px"].astype(float) * scale).astype(int)

    valid = (x >= 0) & (x < display_w) & (y >= 0) & (y < display_h)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return canvas

    density = np.zeros((display_h, display_w), dtype=np.float32)
    np.add.at(density, (y, x), 1)

    if density.max() > 0:
        normalized = (density / density.max() * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        canvas = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return canvas


@st.cache_data
def load_rally_intensity(hip_csv: str, rally_csv: str) -> pd.DataFrame:
    rally_path = Path(rally_csv)
    if rally_path.exists():
        return pd.read_csv(rally_path)
    return build_rally_intensity_dataframe(Path(hip_csv))


@st.cache_data
def load_center_distance_series(hip_csv: str) -> pd.DataFrame:
    center_df, _, _ = build_center_distance_series(Path(hip_csv))
    if center_df.empty:
        return center_df

    center_df = center_df.copy()
    center_df["smoothed_distance"] = (
        center_df["distance_from_center"]
        .rolling(window=ROLLING_WINDOW, min_periods=1, center=True)
        .mean()
    )
    return center_df


@st.cache_data
def load_movement_series(hip_csv: str) -> pd.DataFrame:
    return build_movement_series(Path(hip_csv))


@st.cache_data
def load_position_data(hip_csv: str) -> pd.DataFrame:
    return load_tracking_positions(Path(hip_csv))


@st.cache_data
def load_zone_metrics(hip_csv: str) -> dict:
    return compute_zone_occupancy_metrics(Path(hip_csv))


@st.cache_data
def load_movement_summary(hip_csv: str) -> dict:
    return calculate_movement_metrics(Path(hip_csv))


@st.cache_data
def load_center_summary(hip_csv: str) -> dict:
    return calculate_center_recovery_metrics(Path(hip_csv))


def lookup_value_at_frame(df: pd.DataFrame, frame_idx: int, value_col: str) -> float | None:
    if df.empty or value_col not in df.columns:
        return None

    match = df[df["frame"] == frame_idx]
    if not match.empty:
        return float(match.iloc[0][value_col])

    closest_idx = (df["frame"] - frame_idx).abs().idxmin()
    return float(df.loc[closest_idx, value_col])


def get_rally_intensity_at_frame(rally_df: pd.DataFrame, frame_idx: int) -> float | None:
    if rally_df.empty:
        return None

    window_start = (frame_idx // RALLY_WINDOW_FRAMES) * RALLY_WINDOW_FRAMES
    match = rally_df[rally_df["window_start_frame"] == window_start]
    if match.empty:
        return None

    return float(match.iloc[0]["intensity_score"])


def render_frame_metrics(
    frame_idx: int,
    fps: float,
    movement_df: pd.DataFrame,
    center_df: pd.DataFrame,
    rally_df: pd.DataFrame,
    positions_df: pd.DataFrame,
) -> None:
    st.metric("Current Frame", f"{frame_idx:,}")
    st.metric("Timestamp", f"{frame_idx / fps:.2f} s")

    movement = lookup_value_at_frame(movement_df, frame_idx, "movement")
    center_distance = lookup_value_at_frame(center_df, frame_idx, "distance_from_center")
    intensity = get_rally_intensity_at_frame(rally_df, frame_idx)

    if movement is not None:
        st.metric("Frame Movement", f"{movement:.1f} px")
    else:
        st.write("Frame Movement: N/A")

    if intensity is not None:
        st.metric("Rally Intensity (3s window)", f"{intensity:.1f}")
    else:
        st.write("Rally Intensity (3s window): N/A")

    if center_distance is not None:
        st.metric("Distance From Center", f"{center_distance:.1f} px")
    else:
        st.write("Distance From Center: N/A")

    x_px = lookup_value_at_frame(positions_df, frame_idx, "x_px")
    y_px = lookup_value_at_frame(positions_df, frame_idx, "y_px")
    if x_px is not None and y_px is not None:
        st.metric("Hip Position", f"({x_px:.0f}, {y_px:.0f})")


def main() -> None:
    st.set_page_config(page_title="Badminton Match Timeline", layout="wide")
    st.title("Badminton Match Timeline Dashboard")

    if not HIP_CSV.exists():
        st.error(f"Missing input file: {HIP_CSV}")
        st.stop()

    video_path = resolve_video_path()
    movement = load_movement_summary(str(HIP_CSV))
    center = load_center_summary(str(HIP_CSV))
    zones = load_zone_metrics(str(HIP_CSV))
    rally_df = load_rally_intensity(str(HIP_CSV), str(RALLY_CSV))
    center_df = load_center_distance_series(str(HIP_CSV))
    movement_df = load_movement_series(str(HIP_CSV))
    positions_df = load_position_data(str(HIP_CSV))

    summary_cols = st.columns(4)
    summary_cols[0].metric("Total Distance", f"{movement['total_distance']:.0f} px")
    summary_cols[1].metric("Avg Movement / Frame", f"{movement['average_movement']:.1f} px")
    summary_cols[2].metric("Avg Dist. From Center", f"{center['average_distance']:.1f} px")
    summary_cols[3].metric("Frames Analyzed", f"{movement['frames_analyzed']:,}")

    st.divider()

    st.subheader("Section 1: Synchronized Video & Heatmap Viewer")

    if video_path is None:
        st.warning("No video file found. Place a video at videos/practice.mp4")
        total_frames = int(positions_df["frame"].max()) if not positions_df.empty else 0
        fps = 30.0
        frame_width = int(positions_df["x_px"].max()) + 1 if not positions_df.empty else 1
        frame_height = int(positions_df["y_px"].max()) + 1 if not positions_df.empty else 1
    else:
        video_meta = get_video_metadata(str(video_path))
        total_frames = video_meta["total_frames"]
        fps = video_meta["fps"]
        frame_width = video_meta["width"]
        frame_height = video_meta["height"]

    frame_idx = st.slider("Frame", 0, max(total_frames - 1, 0), 0)
    st.caption(f"Video FPS: {fps:.2f}  |  Total frames: {total_frames:,}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Video Frame**")
        if video_path is None:
            st.write("No video available")
        else:
            frame_image = read_video_frame(str(video_path), frame_idx)
            if frame_image is not None:
                st.image(frame_image, caption=f"Frame {frame_idx}", use_container_width=True)
            else:
                st.error(f"Could not read frame {frame_idx}")

    with col2:
        st.markdown("**Movement Heatmap**")
        heatmap_image = build_movement_heatmap_frame(
            positions_df,
            frame_idx,
            frame_width,
            frame_height,
        )
        st.image(
            heatmap_image,
            caption=f"Cumulative positions through frame {frame_idx}",
            use_container_width=True,
        )

    st.markdown("**Current Frame Metrics**")
    render_frame_metrics(
        frame_idx,
        fps,
        movement_df,
        center_df,
        rally_df,
        positions_df,
    )

    st.divider()

    st.subheader("Section 2: Movement Intensity")
    if rally_df.empty:
        st.warning("No rally intensity data available.")
    else:
        st.line_chart(
            rally_df,
            x="window_start_frame",
            y="intensity_score",
            color="#c0392b",
        )
        st.caption("Intensity score = total movement per 3-second window (90 frames at 30 FPS).")

    st.divider()

    st.subheader("Section 3: Center Recovery")
    if center_df.empty:
        st.warning("No center recovery data available.")
    else:
        st.line_chart(
            center_df,
            x="frame",
            y="smoothed_distance",
            color="#8e44ad",
        )
        recovery_cols = st.columns(2)
        recovery_cols[0].metric("Average Distance", f"{center['average_distance']:.1f} px")
        recovery_cols[1].metric("Maximum Distance", f"{center['maximum_distance']:.1f} px")

    st.divider()

    st.subheader("Section 4: Zone Usage Summary")
    st.caption(ZONE_NOTE)

    zone_cols = st.columns(2)

    vertical_df = pd.DataFrame(
        {
            "Zone": VERTICAL_ZONE_ORDER,
            "Time Spent (%)": [
                zones["top_pct"],
                zones["middle_pct"],
                zones["bottom_pct"],
            ],
        }
    )
    horizontal_df = pd.DataFrame(
        {
            "Zone": ["Left", "Right"],
            "Time Spent (%)": [zones["left_pct"], zones["right_pct"]],
        }
    )

    with zone_cols[0]:
        st.markdown("**Vertical Zones**")
        for zone, key in zip(
            VERTICAL_ZONE_ORDER,
            ["top_pct", "middle_pct", "bottom_pct"],
        ):
            st.write(f"- {zone}: **{zones[key]:.1f}%**")
        st.bar_chart(vertical_df, x="Zone", y="Time Spent (%)", color="#3498db")

    with zone_cols[1]:
        st.markdown("**Horizontal Zones**")
        st.write(f"- Left: **{zones['left_pct']:.1f}%**")
        st.write(f"- Right: **{zones['right_pct']:.1f}%**")
        st.bar_chart(horizontal_df, x="Zone", y="Time Spent (%)", color="#9b59b6")


if __name__ == "__main__":
    main()
