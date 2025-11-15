# Hackathon Challenge: Lights, Camera, Reaction!

Event camera object detection and tracking challenge by Sensofusion.

## Challenge Overview

Build real-time ML models to detect and interpret motion from microsecond-level event camera vision streams:

1. **Simple Task**: Count fan rotations per second
2. **Advanced Tasks**: Track drones, estimate speed, predict motion
3. **Goal**: Most accurate, creative, technically impressive, and real-time capable solution

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
uv sync

# Verify installation
uv run scripts/play_dat.py --help
```

### 2. Download Dataset

Download event camera recordings from:
https://drive.google.com/drive/folders/18ORzE9_aHABYqOHzVdL0GANk_eIMaSuE?usp=sharing

Place `.dat` files in the `data/` directory.

### 3. Run Example Scripts

```bash
# Visualize event data
uv run scripts/play_dat.py data/fan.dat --window 10 --speed 1

# Fan rotation detection
uv run scripts/fan_rotation_detector.py data/fan.dat

# Drone tracking
uv run scripts/drone_tracker.py data/drone.dat

# Advanced motion analysis
uv run scripts/motion_analyzer.py data/drone.dat --export results.json
```

## Solutions Provided

### 1. Fan Rotation Detector (`scripts/fan_rotation_detector.py`)

**Algorithm**: Temporal pattern analysis of event activity

**Features**:
- Real-time rotation counting
- RPS (Rotations Per Second) measurement
- Adaptive peak detection for different speeds
- ROI (Region of Interest) support for focused detection

**Usage**:
```bash
uv run scripts/fan_rotation_detector.py data/fan.dat --window 5 --speed 1
```

**Parameters**:
- `--window`: Window duration in ms (default: 5ms)
- `--speed`: Playback speed multiplier
- `--roi`: Region of interest (x y width height)

**How it works**:
1. Accumulates event counts in temporal windows
2. Detects peaks in activity (blade passing)
3. Measures time between peaks for rotation frequency
4. Filters false positives with threshold and timing constraints

### 2. Drone Tracker (`scripts/drone_tracker.py`)

**Algorithm**: Spatial-temporal clustering with DBSCAN

**Features**:
- Real-time object detection via event clustering
- Trajectory tracking and visualization
- Velocity and speed estimation
- Adaptive clustering for different object sizes

**Usage**:
```bash
uv run scripts/drone_tracker.py data/drone.dat --window 10 --cluster-eps 25
```

**Parameters**:
- `--window`: Window duration in ms (default: 10ms)
- `--cluster-eps`: DBSCAN clustering epsilon (default: 25.0)
- `--min-events`: Minimum events for cluster (default: 10)

**How it works**:
1. Clusters events spatially using DBSCAN
2. Identifies largest cluster as target object
3. Tracks position changes over time
4. Estimates velocity from position deltas
5. Maintains trajectory history for visualization

### 3. Motion Analyzer (`scripts/motion_analyzer.py`)

**Algorithm**: Comprehensive statistical motion analysis

**Features**:
- Multi-scale spatial analysis
- Velocity and acceleration tracking
- Spatial entropy calculation (event distribution)
- Temporal density metrics
- Real-time visualization dashboard
- JSON export for further analysis

**Usage**:
```bash
uv run scripts/motion_analyzer.py data/recording.dat --window 8 --export results.json
```

**Parameters**:
- `--window`: Window duration in ms (default: 8ms)
- `--grid-size`: Grid resolution for spatial analysis (default: 32)
- `--export`: Export metrics to JSON file

**Metrics computed**:
- Event count and density
- Center of mass position
- Velocity magnitude and direction
- Spatial spread (standard deviation)
- Spatial entropy (distribution uniformity)
- Temporal density (events per second)

## Understanding Event Camera Data

### What are Event Cameras?

Event cameras (Dynamic Vision Sensors) work fundamentally different from traditional cameras:

- **Traditional cameras**: Capture full frames at fixed intervals (30/60 fps)
- **Event cameras**: Each pixel independently detects brightness changes

### Event Properties

Each event has 4 components:
1. **x**: Pixel x-coordinate (0-1279 for 1280x720)
2. **y**: Pixel y-coordinate (0-719 for 1280x720)
3. **timestamp**: Time in microseconds (μs precision!)
4. **polarity**: ON (brightness increase) or OFF (brightness decrease)

### Advantages for Motion Detection

- **Microsecond latency**: ~1000x faster than traditional cameras
- **High dynamic range**: Works in bright sunlight and darkness
- **No motion blur**: Events only on change
- **Low data**: Only transmit changes, not full frames
- **High temporal resolution**: Can detect extremely fast motion

## Key Concepts for Solutions

### 1. Temporal Windows

Event streams are continuous, so we process them in windows:
- Short windows (1-5ms): High temporal resolution, fewer events
- Long windows (10-50ms): More events, better for visualization
- Trade-off: Resolution vs. computation

### 2. Spatial Clustering

Events from moving objects cluster spatially:
- Use DBSCAN or similar clustering
- `eps`: Maximum distance between events in same cluster
- `min_samples`: Minimum events to form cluster

### 3. Temporal Patterns

Periodic motion creates temporal patterns:
- Fan blades: Regular peaks in event activity
- Rotating objects: Sinusoidal patterns
- Use autocorrelation or peak detection

### 4. Optical Flow

Events encode motion implicitly:
- ON events: Object moving into brighter region
- OFF events: Object moving into darker region
- Spatial-temporal gradients reveal flow

## Performance Optimization Tips

### 1. Real-Time Processing

Event cameras generate massive data rates (millions of events/second):
- Process events in batches (windows)
- Use vectorized NumPy operations
- Avoid Python loops over events
- Memory-map large files (evio does this automatically)

### 2. Algorithm Efficiency

- **Clustering**: Use spatial indexing (KD-trees) for large event counts
- **Tracking**: Implement predictive tracking (Kalman filters)
- **Detection**: Use hierarchical approaches (coarse-to-fine)

### 3. Parameter Tuning

Each recording may need different parameters:
- Window size: Depends on object speed
- Clustering epsilon: Depends on object size
- Thresholds: Depends on scene lighting/contrast

## Advanced Ideas to Explore

### 1. Learning-Based Approaches

- Train neural networks on event data
- Event-based optical flow networks
- Spiking Neural Networks (SNNs)

### 2. Multi-Object Tracking

- Track multiple drones simultaneously
- Data association across frames
- Handle occlusions and crossings

### 3. Predictive Tracking

- Kalman filtering for smooth trajectories
- Predict future positions
- Handle temporary occlusions

### 4. 3D Reconstruction

- Estimate 3D motion from 2D events
- Structure from motion
- Depth estimation

### 5. Event-Based Features

- Corner detection (eCorner, eFAST)
- Edge detection
- Feature tracking

## Evaluation Criteria

The challenge looks for:

1. **Accuracy**: Correct detection/tracking results
2. **Creativity**: Novel approaches and insights
3. **Technical Quality**: Clean code, good architecture
4. **Real-Time Capability**: Efficient, low-latency processing

## Tips for Success

1. **Start Simple**:
   - First, visualize the data with `play_dat.py`
   - Understand event patterns visually
   - Test basic algorithms before complexity

2. **Iterate Quickly**:
   - Use short recordings for fast iteration
   - Add visualization to debug algorithms
   - Profile code to find bottlenecks

3. **Experiment with Parameters**:
   - Window size drastically affects results
   - Try different clustering parameters
   - Document what works and why

4. **Validate Results**:
   - For fan: Count manually and compare
   - For drone: Verify trajectories make sense
   - Export metrics and analyze statistically

5. **Be Creative**:
   - Combine multiple approaches
   - Use domain knowledge (physics of motion)
   - Try unconventional visualizations

## Resources

### Code Examples
- `scripts/play_dat.py`: Visualization baseline
- `scripts/fan_rotation_detector.py`: Simple periodic detection
- `scripts/drone_tracker.py`: Clustering-based tracking
- `scripts/motion_analyzer.py`: Comprehensive analysis

### Documentation
- `README.md`: Library overview and file format
- `data/README.md`: Dataset information

### External Resources
- Prophesee Metavision SDK: https://docs.prophesee.ai/
- Event Camera Survey: https://arxiv.org/abs/1904.08405
- Event-Based Vision Resources: https://github.com/uzh-rpg/event-based_vision_resources

## Next Steps

1. **Download the dataset** from Google Drive
2. **Run the example scripts** to understand the data
3. **Modify and experiment** with the provided algorithms
4. **Develop your own solution** building on these foundations
5. **Test and validate** on the full dataset
6. **Prepare your demo** showing real-time capabilities

Good luck! Push the limits of microsecond vision! 🚀
