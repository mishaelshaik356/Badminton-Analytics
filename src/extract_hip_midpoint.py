import csv
from pathlib import Path

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose


def hip_midpoint(landmarks, frame_width: int, frame_height: int) -> dict | None:
    """Return the midpoint between left and right hip landmarks."""
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    x_norm = (left_hip.x + right_hip.x) / 2
    y_norm = (left_hip.y + right_hip.y) / 2
    z_norm = (left_hip.z + right_hip.z) / 2
    visibility = (left_hip.visibility + right_hip.visibility) / 2

    return {
        "x_norm": x_norm,
        "y_norm": y_norm,
        "z_norm": z_norm,
        "x_px": x_norm * frame_width,
        "y_px": y_norm * frame_height,
        "visibility": visibility,
    }


def extract_hip_midpoints(video_path: str | Path, output_csv: str | Path) -> int:
    """
    Process a video and save hip midpoint coordinates for every frame to CSV.

    Returns the number of frames written.
    """
    video_path = Path(video_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fieldnames = [
        "frame",
        "timestamp_sec",
        "detected",
        "x_norm",
        "y_norm",
        "z_norm",
        "x_px",
        "y_px",
        "visibility",
    ]

    frame_index = 0

    with mp_pose.Pose() as pose, output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            row = {
                "frame": frame_index,
                "timestamp_sec": round(frame_index / fps, 4),
                "detected": False,
                "x_norm": "",
                "y_norm": "",
                "z_norm": "",
                "x_px": "",
                "y_px": "",
                "visibility": "",
            }

            if results.pose_landmarks:
                midpoint = hip_midpoint(
                    results.pose_landmarks.landmark,
                    frame_width,
                    frame_height,
                )
                row.update(
                    {
                        "detected": True,
                        "x_norm": round(midpoint["x_norm"], 6),
                        "y_norm": round(midpoint["y_norm"], 6),
                        "z_norm": round(midpoint["z_norm"], 6),
                        "x_px": round(midpoint["x_px"], 2),
                        "y_px": round(midpoint["y_px"], 2),
                        "visibility": round(midpoint["visibility"], 4),
                    }
                )

            writer.writerow(row)
            frame_index += 1

    cap.release()
    return frame_index


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    video = project_root / "videos" / "practice.mp4"
    output = project_root / "output" / "hip_midpoints.csv"

    frames_written = extract_hip_midpoints(video, output)
    print(f"Saved {frames_written} frames to {output}")
