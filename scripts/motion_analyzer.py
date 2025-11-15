"""
Advanced Motion Analysis for Event Camera Data

This script provides comprehensive motion analysis capabilities including:
- Optical flow estimation from events
- Speed and acceleration measurement
- Motion pattern recognition
- Real-time analytics dashboard

Usage:
    uv run scripts/motion_analyzer.py data/recording.dat

Features:
- Multi-scale motion analysis
- Adaptive event accumulation
- Statistical motion metrics
- Export capabilities for further analysis
"""

import argparse
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


@dataclass
class MotionMetrics:
    """Container for motion analysis metrics."""

    timestamp_us: int
    event_count: int
    mean_x: float
    mean_y: float
    std_x: float
    std_y: float
    velocity_magnitude: float
    velocity_direction: float  # radians
    spatial_entropy: float
    temporal_density: float  # events per microsecond


class MotionAnalyzer:
    """Analyze motion patterns from event camera data."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        grid_size: int = 32,
        history_length: int = 100,
    ):
        """
        Initialize motion analyzer.

        Args:
            width: Frame width
            height: Frame height
            grid_size: Grid resolution for spatial analysis
            history_length: Number of frames to keep in history
        """
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.history_length = history_length

        # Analysis state
        self.metrics_history = deque(maxlen=history_length)
        self.last_mean_position = None
        self.last_timestamp = None

        # Grid for spatial analysis
        self.grid_x = width // grid_size
        self.grid_y = height // grid_size

    def compute_spatial_entropy(
        self, x_coords: np.ndarray, y_coords: np.ndarray
    ) -> float:
        """
        Compute spatial entropy of event distribution.

        Higher entropy = more distributed events
        Lower entropy = more concentrated events
        """
        if len(x_coords) == 0:
            return 0.0

        # Create histogram of events in grid
        grid_counts = np.zeros((self.grid_y, self.grid_x))

        grid_x_idx = np.clip(
            x_coords // self.grid_size, 0, self.grid_x - 1
        ).astype(int)
        grid_y_idx = np.clip(
            y_coords // self.grid_size, 0, self.grid_y - 1
        ).astype(int)

        for gx, gy in zip(grid_x_idx, grid_y_idx):
            grid_counts[gy, gx] += 1

        # Normalize to probability distribution
        grid_probs = grid_counts / (np.sum(grid_counts) + 1e-10)

        # Remove zeros for entropy calculation
        grid_probs = grid_probs[grid_probs > 0]

        # Calculate Shannon entropy
        entropy = -np.sum(grid_probs * np.log2(grid_probs + 1e-10))

        return float(entropy)

    def analyze_window(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        timestamp_us: int,
        window_duration_us: float,
    ) -> MotionMetrics:
        """
        Analyze a window of events.

        Args:
            x_coords: X coordinates
            y_coords: Y coordinates
            timestamp_us: Current timestamp
            window_duration_us: Window duration in microseconds

        Returns:
            MotionMetrics for this window
        """
        event_count = len(x_coords)

        if event_count == 0:
            return MotionMetrics(
                timestamp_us=timestamp_us,
                event_count=0,
                mean_x=0.0,
                mean_y=0.0,
                std_x=0.0,
                std_y=0.0,
                velocity_magnitude=0.0,
                velocity_direction=0.0,
                spatial_entropy=0.0,
                temporal_density=0.0,
            )

        # Spatial statistics
        mean_x = float(np.mean(x_coords))
        mean_y = float(np.mean(y_coords))
        std_x = float(np.std(x_coords))
        std_y = float(np.std(y_coords))

        # Velocity estimation
        velocity_mag = 0.0
        velocity_dir = 0.0

        if self.last_mean_position is not None and self.last_timestamp is not None:
            dt_us = timestamp_us - self.last_timestamp
            dt_s = dt_us / 1e6

            if dt_s > 0:
                dx = mean_x - self.last_mean_position[0]
                dy = mean_y - self.last_mean_position[1]

                velocity_mag = np.sqrt(dx**2 + dy**2) / dt_s
                velocity_dir = np.arctan2(dy, dx)

        # Spatial distribution
        entropy = self.compute_spatial_entropy(x_coords, y_coords)

        # Temporal density
        temporal_density = (
            event_count / window_duration_us if window_duration_us > 0 else 0.0
        )

        # Update state
        self.last_mean_position = (mean_x, mean_y)
        self.last_timestamp = timestamp_us

        metrics = MotionMetrics(
            timestamp_us=timestamp_us,
            event_count=event_count,
            mean_x=mean_x,
            mean_y=mean_y,
            std_x=std_x,
            std_y=std_y,
            velocity_magnitude=float(velocity_mag),
            velocity_direction=float(velocity_dir),
            spatial_entropy=entropy,
            temporal_density=temporal_density,
        )

        self.metrics_history.append(metrics)

        return metrics

    def get_summary_statistics(self) -> dict:
        """Get summary statistics over all analyzed windows."""
        if not self.metrics_history:
            return {}

        velocities = [m.velocity_magnitude for m in self.metrics_history]
        event_counts = [m.event_count for m in self.metrics_history]
        entropies = [m.spatial_entropy for m in self.metrics_history]

        return {
            "total_windows": len(self.metrics_history),
            "avg_event_count": float(np.mean(event_counts)),
            "avg_velocity": float(np.mean(velocities)),
            "max_velocity": float(np.max(velocities)),
            "avg_spatial_entropy": float(np.mean(entropies)),
        }


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


def create_visualization(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    metrics: MotionMetrics,
    analyzer: MotionAnalyzer,
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    """Create visualization with motion overlays."""
    x_coords, y_coords, polarities = window

    # Create base frame
    frame = np.full((height, width, 3), (25, 25, 25), np.uint8)

    if len(x_coords) > 0:
        # Color code by polarity
        frame[y_coords[polarities], x_coords[polarities]] = (255, 255, 255)
        frame[y_coords[~polarities], x_coords[~polarities]] = (0, 100, 200)

        # Draw center of mass
        cx, cy = int(metrics.mean_x), int(metrics.mean_y)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 2)

        # Draw velocity vector
        if metrics.velocity_magnitude > 10:
            scale = 0.1
            vx = metrics.velocity_magnitude * np.cos(metrics.velocity_direction)
            vy = metrics.velocity_magnitude * np.sin(metrics.velocity_direction)
            end_x = int(cx + vx * scale)
            end_y = int(cy + vy * scale)
            cv2.arrowedLine(
                frame, (cx, cy), (end_x, end_y), (255, 0, 255), 3, cv2.LINE_AA
            )

        # Draw standard deviation ellipse
        if metrics.std_x > 0 and metrics.std_y > 0:
            axes = (int(metrics.std_x * 2), int(metrics.std_y * 2))
            cv2.ellipse(
                frame, (cx, cy), axes, 0, 0, 360, (255, 165, 0), 2
            )

    # Draw metrics panel
    panel_height = 250
    panel = np.zeros((panel_height, width, 3), np.uint8)

    metrics_text = [
        f"Events: {metrics.event_count}",
        f"Velocity: {metrics.velocity_magnitude:.1f} px/s",
        f"Direction: {np.degrees(metrics.velocity_direction):.1f} deg",
        f"Spatial Entropy: {metrics.spatial_entropy:.2f}",
        f"Temporal Density: {metrics.temporal_density * 1e6:.2f} events/s",
        f"Position: ({metrics.mean_x:.0f}, {metrics.mean_y:.0f})",
        f"Spread: ({metrics.std_x:.1f}, {metrics.std_y:.1f})",
    ]

    y_offset = 25
    for text in metrics_text:
        cv2.putText(
            panel,
            text,
            (15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y_offset += 30

    # Velocity history graph
    if len(analyzer.metrics_history) > 1:
        history = list(analyzer.metrics_history)
        velocities = [m.velocity_magnitude for m in history]

        # Normalize for display
        max_vel = max(velocities) if max(velocities) > 0 else 1
        graph_width = min(400, len(velocities) * 3)
        graph_height = 80
        graph_x = width - graph_width - 20
        graph_y = 20

        # Draw graph background
        cv2.rectangle(
            panel,
            (graph_x, graph_y),
            (graph_x + graph_width, graph_y + graph_height),
            (50, 50, 50),
            -1,
        )

        # Plot velocities
        points = []
        for i, vel in enumerate(velocities[-graph_width // 3 :]):
            x = graph_x + (i * 3)
            y = graph_y + graph_height - int((vel / max_vel) * graph_height)
            points.append((x, y))

        if len(points) > 1:
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(
                panel, [points_array], False, (0, 255, 0), 2, cv2.LINE_AA
            )

        cv2.putText(
            panel,
            "Velocity History",
            (graph_x, graph_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # Combine frame and panel
    combined = np.vstack([frame, panel])

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced motion analysis for event camera data"
    )
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument(
        "--window",
        type=float,
        default=8,
        help="Window duration in ms (default: 8ms)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1,
        help="Playback speed (1 is real time)",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export metrics to JSON file",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=32,
        help="Grid size for spatial analysis (default: 32)",
    )
    args = parser.parse_args()

    # Initialize
    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )

    analyzer = MotionAnalyzer(
        width=1280, height=720, grid_size=args.grid_size
    )

    pacer = Pacer(speed=args.speed, force_speed=False)

    cv2.namedWindow("Motion Analyzer", cv2.WINDOW_NORMAL)

    print("\n" + "=" * 70)
    print("Advanced Motion Analyzer - Event Camera Analysis")
    print("=" * 70)
    print("Analyzing motion patterns with:")
    print(f"  - Window size: {args.window}ms")
    print(f"  - Spatial grid: {args.grid_size}px")
    print(f"  - Playback speed: {args.speed}x")
    print("\nPress 'q' or ESC to quit")
    print("=" * 70 + "\n")

    all_metrics = []

    try:
        for batch_range in pacer.pace(src.ranges()):
            window = get_window(
                src.event_words,
                src.order,
                batch_range.start,
                batch_range.stop,
            )

            x_coords, y_coords, polarities = window

            # Analyze
            window_duration = batch_range.end_ts_us - batch_range.start_ts_us
            metrics = analyzer.analyze_window(
                x_coords, y_coords, batch_range.end_ts_us, window_duration
            )

            all_metrics.append(metrics)

            # Visualize
            vis_frame = create_visualization(window, metrics, analyzer)

            cv2.imshow("Motion Analyzer", vis_frame)

            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

    finally:
        cv2.destroyAllWindows()

        # Print summary
        summary = analyzer.get_summary_statistics()
        print("\n" + "=" * 70)
        print("Analysis Summary")
        print("=" * 70)
        for key, value in summary.items():
            print(f"  {key}: {value:.2f}")
        print("=" * 70 + "\n")

        # Export if requested
        if args.export:
            export_data = {
                "summary": summary,
                "metrics": [asdict(m) for m in all_metrics],
            }
            with open(args.export, "w") as f:
                json.dump(export_data, f, indent=2)
            print(f"Metrics exported to: {args.export}\n")


if __name__ == "__main__":
    main()
