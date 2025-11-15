"""
Real-Time Drone Tracking using Event Camera Data

This script implements advanced object tracking for fast-moving drones using
event camera data. It uses spatial-temporal clustering to detect the drone
and tracks its motion with speed estimation.

Usage:
    uv run scripts/drone_tracker.py data/drone.dat

Features:
- Real-time object detection via event clustering
- Motion tracking with trajectory visualization
- Speed and velocity estimation
- Adaptive clustering for different drone sizes
- Low-latency processing optimized for event cameras
"""

import argparse
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


@dataclass
class TrackedObject:
    """Represents a tracked object with position and velocity."""

    position: tuple[float, float]  # (x, y) in pixels
    velocity: tuple[float, float]  # (vx, vy) in pixels/second
    timestamp_us: int
    confidence: float
    size: int  # Number of events in cluster


class DroneTracker:
    """Track fast-moving objects using event clustering."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        cluster_eps: float = 25.0,
        min_cluster_size: int = 10,
        max_history: int = 50,
    ):
        """
        Initialize drone tracker.

        Args:
            width: Frame width
            height: Frame height
            cluster_eps: DBSCAN epsilon for clustering events
            min_cluster_size: Minimum events to form a cluster
            max_history: Maximum trajectory history to keep
        """
        self.width = width
        self.height = height
        self.cluster_eps = cluster_eps
        self.min_cluster_size = min_cluster_size

        # Tracking state
        self.trajectory = deque(maxlen=max_history)
        self.last_position = None
        self.last_timestamp = None

        # Statistics
        self.total_detections = 0
        self.speed_history = deque(maxlen=100)

    def cluster_events(
        self, x_coords: np.ndarray, y_coords: np.ndarray
    ) -> list[tuple[float, float, int]]:
        """
        Cluster events using DBSCAN to find distinct objects.

        Args:
            x_coords: X coordinates of events
            y_coords: Y coordinates of events

        Returns:
            List of (center_x, center_y, cluster_size) for each detected cluster
        """
        if len(x_coords) < self.min_cluster_size:
            return []

        # Stack coordinates for clustering
        coords = np.column_stack([x_coords, y_coords])

        # DBSCAN clustering
        clustering = DBSCAN(
            eps=self.cluster_eps, min_samples=self.min_cluster_size
        ).fit(coords)

        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        clusters = []
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_x = x_coords[cluster_mask]
            cluster_y = y_coords[cluster_mask]

            # Compute cluster center
            center_x = float(np.mean(cluster_x))
            center_y = float(np.mean(cluster_y))
            size = int(np.sum(cluster_mask))

            clusters.append((center_x, center_y, size))

        # Sort by size (largest first)
        clusters.sort(key=lambda c: c[2], reverse=True)

        return clusters

    def update_tracking(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        timestamp_us: int,
    ) -> TrackedObject | None:
        """
        Update object tracking with new events.

        Args:
            x_coords: X coordinates of events
            y_coords: Y coordinates of events
            timestamp_us: Current timestamp in microseconds

        Returns:
            TrackedObject if detected, None otherwise
        """
        # Cluster events to find objects
        clusters = self.cluster_events(x_coords, y_coords)

        if not clusters:
            return None

        # Take largest cluster as the drone
        center_x, center_y, size = clusters[0]

        # Calculate velocity if we have previous position
        velocity = (0.0, 0.0)
        if self.last_position is not None and self.last_timestamp is not None:
            dt_us = timestamp_us - self.last_timestamp
            dt_s = dt_us / 1e6

            if dt_s > 0:
                dx = center_x - self.last_position[0]
                dy = center_y - self.last_position[1]
                vx = dx / dt_s
                vy = dy / dt_s
                velocity = (vx, vy)

                # Track speed
                speed = np.sqrt(vx**2 + vy**2)
                self.speed_history.append(speed)

        # Update state
        self.last_position = (center_x, center_y)
        self.last_timestamp = timestamp_us
        self.total_detections += 1

        # Calculate confidence based on cluster size
        confidence = min(1.0, size / 100.0)

        tracked = TrackedObject(
            position=(center_x, center_y),
            velocity=velocity,
            timestamp_us=timestamp_us,
            confidence=confidence,
            size=size,
        )

        # Add to trajectory
        self.trajectory.append((center_x, center_y))

        return tracked

    def get_average_speed(self) -> float:
        """Get average speed in pixels/second."""
        if not self.speed_history:
            return 0.0
        return float(np.mean(self.speed_history))

    def get_trajectory_array(self) -> np.ndarray:
        """Get trajectory as numpy array for visualization."""
        if len(self.trajectory) < 2:
            return np.array([]).reshape(0, 2)
        return np.array(self.trajectory, dtype=np.int32)


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract event data for a time window."""
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (20, 20, 20),
    on_color: tuple[int, int, int] = (255, 255, 255),
    off_color: tuple[int, int, int] = (0, 80, 200),
) -> np.ndarray:
    """Render events as frame."""
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)

    if len(x_coords) > 0:
        frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
        frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def draw_tracking_info(
    frame: np.ndarray,
    tracked: TrackedObject | None,
    tracker: DroneTracker,
    batch_range: BatchRange,
) -> None:
    """Draw tracking visualization and HUD."""
    # Draw trajectory
    trajectory = tracker.get_trajectory_array()
    if len(trajectory) >= 2:
        cv2.polylines(
            frame, [trajectory], False, (0, 255, 0), 2, cv2.LINE_AA
        )

    # Draw current detection
    if tracked is not None:
        x, y = tracked.position
        x, y = int(x), int(y)

        # Draw center point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw bounding circle
        radius = max(10, int(tracked.size / 5))
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)

        # Draw velocity vector
        vx, vy = tracked.velocity
        if abs(vx) > 1 or abs(vy) > 1:
            # Scale velocity for visualization
            scale = 0.05
            end_x = int(x + vx * scale)
            end_y = int(y + vy * scale)
            cv2.arrowedLine(
                frame, (x, y), (end_x, end_y), (255, 0, 255), 2, cv2.LINE_AA
            )

        # Speed info
        speed = np.sqrt(vx**2 + vy**2)
        cv2.putText(
            frame,
            f"Speed: {speed:.1f} px/s",
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Position info
        cv2.putText(
            frame,
            f"({x}, {y})",
            (x + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # HUD - General info
    rec_time_s = batch_range.end_ts_us / 1e6

    cv2.putText(
        frame,
        f"Time: {rec_time_s:.2f}s",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Detections: {tracker.total_detections}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    avg_speed = tracker.get_average_speed()
    cv2.putText(
        frame,
        f"Avg Speed: {avg_speed:.1f} px/s",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Status indicator
    status = "TRACKING" if tracked is not None else "SEARCHING"
    color = (0, 255, 0) if tracked is not None else (0, 165, 255)
    cv2.putText(
        frame,
        status,
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track drones from event camera data"
    )
    parser.add_argument("dat", help="Path to .dat file with drone recording")
    parser.add_argument(
        "--window",
        type=float,
        default=10,
        help="Window duration in ms (default: 10ms)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1,
        help="Playback speed (1 is real time)",
    )
    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=25.0,
        help="DBSCAN clustering epsilon (default: 25.0)",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=10,
        help="Minimum events for cluster (default: 10)",
    )
    args = parser.parse_args()

    # Initialize source and tracker
    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )

    tracker = DroneTracker(
        width=1280,
        height=720,
        cluster_eps=args.cluster_eps,
        min_cluster_size=args.min_events,
    )

    pacer = Pacer(speed=args.speed, force_speed=False)

    cv2.namedWindow("Drone Tracker", cv2.WINDOW_NORMAL)

    print("\n" + "=" * 60)
    print("Drone Tracker - Real-Time Event Camera Tracking")
    print("=" * 60)
    print("Controls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Adjust --cluster-eps for different object sizes")
    print("  - Use --window to change temporal resolution")
    print("=" * 60 + "\n")

    for batch_range in pacer.pace(src.ranges()):
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )

        x_coords, y_coords, polarities = window

        # Update tracking
        tracked = tracker.update_tracking(
            x_coords, y_coords, batch_range.end_ts_us
        )

        # Visualize
        frame = get_frame(window)
        draw_tracking_info(frame, tracked, tracker, batch_range)

        cv2.imshow("Drone Tracker", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

    cv2.destroyAllWindows()

    # Final summary
    print("\n" + "=" * 60)
    print("Tracking Summary")
    print("=" * 60)
    print(f"Total Detections: {tracker.total_detections}")
    if tracker.speed_history:
        avg_speed = tracker.get_average_speed()
        max_speed = max(tracker.speed_history)
        print(f"Average Speed: {avg_speed:.2f} px/s")
        print(f"Maximum Speed: {max_speed:.2f} px/s")
    print(f"Trajectory Points: {len(tracker.trajectory)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
