"""
Performance Benchmark for Event Camera Processing

This script benchmarks the real-time capabilities of event processing algorithms.
Measures throughput, latency, and computational efficiency.

Usage:
    uv run scripts/benchmark.py data/recording.dat

Features:
- Processing throughput (events/second)
- Average latency per window
- Memory usage tracking
- Real-time performance metrics
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from evio.source.dat_file import DatFileSource


@dataclass
class BenchmarkStats:
    """Container for benchmark statistics."""

    total_events: int
    total_windows: int
    total_time_s: float
    events_per_second: float
    windows_per_second: float
    avg_window_time_ms: float
    min_window_time_ms: float
    max_window_time_ms: float
    recording_duration_s: float
    real_time_factor: float  # How much faster/slower than real-time


def simple_processing(
    x_coords: np.ndarray, y_coords: np.ndarray, polarities: np.ndarray
) -> dict:
    """
    Simple processing function for benchmarking.

    Replace this with your own algorithm to benchmark it.
    """
    return {
        "count": len(x_coords),
        "mean_x": float(np.mean(x_coords)) if len(x_coords) > 0 else 0.0,
        "mean_y": float(np.mean(y_coords)) if len(y_coords) > 0 else 0.0,
    }


def extract_events(
    event_words: np.ndarray, time_order: np.ndarray, start: int, stop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and decode events for a window."""
    event_indexes = time_order[start:stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def run_benchmark(
    dat_path: str, window_ms: float, max_windows: int | None = None
) -> BenchmarkStats:
    """
    Run benchmark on event camera recording.

    Args:
        dat_path: Path to .dat file
        window_ms: Window duration in milliseconds
        max_windows: Maximum windows to process (None = all)

    Returns:
        BenchmarkStats with performance metrics
    """
    print(f"\nLoading: {dat_path}")
    print(f"Window size: {window_ms}ms")

    src = DatFileSource(
        dat_path, width=1280, height=720, window_length_us=window_ms * 1000
    )

    print(f"Total windows: {len(src)}")

    window_times = []
    total_events = 0

    start_time = time.perf_counter()
    first_timestamp = None
    last_timestamp = None

    windows_processed = 0

    print("\nProcessing windows...")

    for i, batch_range in enumerate(src.ranges()):
        if max_windows is not None and i >= max_windows:
            break

        # Track timestamps
        if first_timestamp is None:
            first_timestamp = batch_range.start_ts_us
        last_timestamp = batch_range.end_ts_us

        # Extract events
        window_start = time.perf_counter()

        x_coords, y_coords, polarities = extract_events(
            src.event_words, src.order, batch_range.start, batch_range.stop
        )

        # Process (this is what you'd replace with your algorithm)
        result = simple_processing(x_coords, y_coords, polarities)

        window_end = time.perf_counter()

        # Track stats
        window_time = (window_end - window_start) * 1000  # ms
        window_times.append(window_time)
        total_events += len(x_coords)
        windows_processed += 1

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} windows...")

    end_time = time.perf_counter()

    # Calculate statistics
    total_time = end_time - start_time
    recording_duration = (last_timestamp - first_timestamp) / 1e6  # seconds

    stats = BenchmarkStats(
        total_events=total_events,
        total_windows=windows_processed,
        total_time_s=total_time,
        events_per_second=total_events / total_time if total_time > 0 else 0,
        windows_per_second=windows_processed / total_time if total_time > 0 else 0,
        avg_window_time_ms=float(np.mean(window_times)),
        min_window_time_ms=float(np.min(window_times)),
        max_window_time_ms=float(np.max(window_times)),
        recording_duration_s=recording_duration,
        real_time_factor=recording_duration / total_time if total_time > 0 else 0,
    )

    return stats


def print_stats(stats: BenchmarkStats) -> None:
    """Print benchmark statistics in a formatted way."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print("\nData Statistics:")
    print(f"  Total Events Processed: {stats.total_events:,}")
    print(f"  Total Windows Processed: {stats.total_windows:,}")
    print(f"  Recording Duration: {stats.recording_duration_s:.3f}s")

    print("\nProcessing Performance:")
    print(f"  Total Processing Time: {stats.total_time_s:.3f}s")
    print(f"  Events/Second: {stats.events_per_second:,.0f}")
    print(f"  Windows/Second: {stats.windows_per_second:.1f}")

    print("\nLatency Metrics:")
    print(f"  Avg Window Time: {stats.avg_window_time_ms:.3f}ms")
    print(f"  Min Window Time: {stats.min_window_time_ms:.3f}ms")
    print(f"  Max Window Time: {stats.max_window_time_ms:.3f}ms")

    print("\nReal-Time Performance:")
    print(f"  Real-Time Factor: {stats.real_time_factor:.2f}x")

    if stats.real_time_factor >= 1.0:
        print(f"  ✓ REAL-TIME CAPABLE (processing {stats.real_time_factor:.1f}x faster than recording)")
    else:
        speedup_needed = 1.0 / stats.real_time_factor
        print(f"  ✗ NOT REAL-TIME (need {speedup_needed:.1f}x speedup)")

    print("\nThroughput:")
    events_per_ms = stats.events_per_second / 1000
    print(f"  Events/ms: {events_per_ms:,.1f}")

    # Estimate for different event rates
    print("\nProjected Performance at Different Event Rates:")
    for rate_meps in [0.5, 1.0, 2.0, 5.0, 10.0]:  # Mega-events per second
        rate_eps = rate_meps * 1e6
        can_handle = stats.events_per_second >= rate_eps
        status = "✓" if can_handle else "✗"
        print(f"  {status} {rate_meps:.1f} MEPS: {'YES' if can_handle else 'NO'}")

    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark event camera processing performance"
    )
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument(
        "--window",
        type=float,
        default=10,
        help="Window duration in ms (default: 10ms)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Maximum windows to process (default: all)",
    )
    args = parser.parse_args()

    # Run benchmark
    stats = run_benchmark(args.dat, args.window, args.max_windows)

    # Print results
    print_stats(stats)

    # Additional recommendations
    print("Optimization Tips:")
    print("  1. Use NumPy vectorized operations (avoid Python loops)")
    print("  2. Process events in batches/windows")
    print("  3. Use memory-mapped files for large recordings")
    print("  4. Consider multi-threading for independent windows")
    print("  5. Profile your code to find bottlenecks")
    print()


if __name__ == "__main__":
    main()
