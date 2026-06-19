from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_tracking_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("frame").reset_index(drop=True)
    return df


def get_point_for_frame(df: pd.DataFrame, frame_idx: int) -> tuple[int, int] | None:
    """Return pixel coordinates for a frame, using the closest CSV row if needed."""
    match = df[df["frame"] == frame_idx]
    row = match.iloc[0] if not match.empty else df.iloc[(df["frame"] - frame_idx).abs().argmin()]

    if pd.isna(row["x_px"]) or pd.isna(row["y_px"]):
        return None

    return int(row["x_px"]), int(row["y_px"])


def run_sync_viewer(video_path: Path, csv_path: Path) -> None:
    df = load_tracking_data(csv_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay_ms = max(1, int(1000 / fps))

    density = np.zeros((height, width), dtype=np.float32)
    dots_panel = np.zeros((height, width, 3), dtype=np.uint8)
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        point = get_point_for_frame(df, frame_idx)
        if point is not None:
            x, y = point
            if 0 <= x < width and 0 <= y < height:
                density[y, x] += 1
                cv2.circle(dots_panel, (x, y), 4, (0, 0, 255), -1)

        heatmap_panel = np.zeros((height, width, 3), dtype=np.uint8)
        if density.max() > 0:
            normalized = (density / density.max() * 255).astype(np.uint8)
            heatmap_panel = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        red_pixels = dots_panel[:, :, 2] > 0
        heatmap_panel[red_pixels] = dots_panel[red_pixels]

        combined = np.hstack([frame, heatmap_panel])

        cv2.putText(
            combined,
            f"Frame {frame_idx} | Q=quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Badminton Sync Viewer", combined)

        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    video = project_root / "videos" / "practice.mp4"
    csv_file = project_root / "output" / "hip_midpoints.csv"

    run_sync_viewer(video, csv_file)
