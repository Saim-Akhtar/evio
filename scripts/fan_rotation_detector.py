"""
Fan Rotation Detection using Event Camera Data

This script detects and counts fan blade rotations from event camera recordings.
The algorithm uses temporal pattern analysis of event activity to identify periodic
motion characteristic of rotating fan blades.

Usage:
    uv run scripts/fan_rotation_detector.py data/fan.dat

Features:
- Real-time rotation detection
- Rotations per second (RPS) measurement
- Visual feedback with event accumulation
- Adaptive thresholding for different speeds
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
from scipy import signal

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


class FanRotationDetector:
    """Detect fan rotations using temporal event patterns."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        history_duration_ms: float = 500,
        detection_roi: tuple[int, int, int, int] | None = None,
    ):
        """
        Initialize rotation detector.

        Args:
            width: Frame width
            height: Frame height
            history_duration_ms: How long to track event history (ms)
            detection_roi: Region of interest (x, y, w, h) for detection
        """
        self.width = width
        self.height = height
        self.history_duration_ms = history_duration_ms

        # ROI for detection (None = full frame)
        self.roi = detection_roi

        # Temporal activity tracking
        self.event_counts = deque(maxlen=1000)
        self.event_timestamps = deque(maxlen=1000)

        # Rotation counting
        self.rotation_count = 0
        self.last_peak_time = None
        self.rps_history = deque(maxlen=20)

        # For peak detection
        self.peak_threshold = 1.5  # Activity must be 1.5x average to count as peak

    def get_roi_mask(self) -> np.ndarray | None:
        """Create binary mask for ROI."""
        if self.roi is None:
            return None

        mask = np.zeros((self.height, self.width), dtype=bool)
        x, y, w, h = self.roi
        mask[y : y + h, x : x + w] = True
        return mask

    def process_window(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        polarities: np.ndarray,
        timestamp_us: int,
    ) -> tuple[float, float]:
        """
        Process event window and detect rotations.

        Args:
            x_coords: X coordinates of events
            y_coords: Y coordinates of events
            polarities: Event polarities
            timestamp_us: Current timestamp in microseconds

        Returns:
            (current_rps, total_rotations)
        """
        # Apply ROI if specified
        if self.roi is not None:
            x, y, w, h = self.roi
            in_roi = (
                (x_coords >= x)
                & (x_coords < x + w)
                & (y_coords >= y)
                & (y_coords < y + h)
            )
            event_count = np.sum(in_roi)
        else:
            event_count = len(x_coords)

        # Track temporal event activity
        self.event_counts.append(event_count)
        self.event_timestamps.append(timestamp_us)

        # Need enough history for detection
        if len(self.event_counts) < 20:
            return 0.0, float(self.rotation_count)

        # Convert to numpy for analysis
        counts = np.array(self.event_counts)
        timestamps = np.array(self.event_timestamps)

        # Detect peaks in event activity (indicates blade passing)
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        # Simple peak detection: current activity significantly above average
        if event_count > mean_count + self.peak_threshold * std_count:
            # Check if enough time passed since last peak
            if self.last_peak_time is not None:
                time_diff_us = timestamp_us - self.last_peak_time
                time_diff_s = time_diff_us / 1e6

                # Reasonable rotation period: 10ms to 1000ms (0.01 to 1 second)
                if 0.01 < time_diff_s < 1.0:
                    self.rotation_count += 1
                    rps = 1.0 / time_diff_s
                    self.rps_history.append(rps)

            self.last_peak_time = timestamp_us

        # Calculate current RPS from recent history
        current_rps = np.median(self.rps_history) if self.rps_history else 0.0

        return current_rps, float(self.rotation_count)

    def get_activity_signal(self) -> tuple[np.ndarray, np.ndarray]:
        """Get event activity signal for visualization."""
        if len(self.event_counts) == 0:
            return np.array([]), np.array([])

        counts = np.array(self.event_counts)
        timestamps = np.array(self.event_timestamps)
        return timestamps, counts


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
    base_color: tuple[int, int, int] = (30, 30, 30),
    on_color: tuple[int, int, int] = (255, 255, 255),
    off_color: tuple[int, int, int] = (0, 100, 255),
) -> np.ndarray:
    """Render events as frame."""
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def draw_hud(
    frame: np.ndarray,
    rps: float,
    total_rotations: float,
    batch_range: BatchRange,
) -> None:
    """Draw HUD with rotation information."""
    rec_time_s = batch_range.end_ts_us / 1e6

    # Main rotation info
    cv2.putText(
        frame,
        f"RPS: {rps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Total Rotations: {total_rotations:.0f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Time: {rec_time_s:.2f}s",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect fan rotations from event camera data"
    )
    parser.add_argument("dat", help="Path to .dat file with fan recording")
    parser.add_argument(
        "--window",
        type=float,
        default=5,
        help="Window duration in ms (default: 5ms)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1,
        help="Playback speed (1 is real time)",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Region of interest for detection (x y width height)",
    )
    args = parser.parse_args()

    # Initialize source and detector
    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )

    roi = tuple(args.roi) if args.roi else None
    detector = FanRotationDetector(
        width=1280, height=720, detection_roi=roi
    )

    pacer = Pacer(speed=args.speed, force_speed=False)

    cv2.namedWindow("Fan Rotation Detector", cv2.WINDOW_NORMAL)

    print("\n" + "=" * 60)
    print("Fan Rotation Detector")
    print("=" * 60)
    print("Controls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Adjust --window for different time resolutions")
    print("  - Use --roi to focus on specific area")
    print("=" * 60 + "\n")

    for batch_range in pacer.pace(src.ranges()):
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )

        x_coords, y_coords, polarities = window

        # Detect rotations
        rps, total_rots = detector.process_window(
            x_coords, y_coords, polarities, batch_range.end_ts_us
        )

        # Visualize
        frame = get_frame(window)

        # Draw ROI if specified
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        draw_hud(frame, rps, total_rots, batch_range)

        cv2.imshow("Fan Rotation Detector", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

    cv2.destroyAllWindows()

    # Final summary
    print("\n" + "=" * 60)
    print("Detection Summary")
    print("=" * 60)
    print(f"Total Rotations Detected: {detector.rotation_count}")
    if detector.rps_history:
        avg_rps = np.mean(detector.rps_history)
        print(f"Average RPS: {avg_rps:.2f}")
        print(f"Average RPM: {avg_rps * 60:.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
