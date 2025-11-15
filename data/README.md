# Event Camera Dataset

This directory contains event camera recordings (.dat files) for the hackathon challenge.

## Download Dataset

Download the dataset from the Google Drive link:
https://drive.google.com/drive/folders/18ORzE9_aHABYqOHzVdL0GANk_eIMaSuE?usp=sharing

## Dataset Scenarios

The dataset includes event data of the following scenarios:

### Rotating Fan
Under the `fan` folder you find three distinct scenarios:
-  `fan_const_rpm` is event data produced by a fan rotating at constant speed, duration ~ 10 s, rpm ~ 1100
- `fan_varying_rpm` is event data produced by a fan which changes its rotating speed during the clip, duration ~ 20 s, rpm ∈ [1100, 1300]
- `fan_varying_rpm_turning` is event data produced by a fan which changes its rotating speed and orientation w.r.t the camera during the clip, duration ~ 25 s, rpm ∈ [1100, 1300]

### Drone Idle
`drone_idle` is event data recorded from a drone hovering stationary at roughly 100 m away from the camera with a tree wobbling on the background, duration ~ 10 s, rpm ∈ [5000, 6000]

### Drone Moving
`drone_moving` is event data recorded from a drone moving around at roughly 100 m away from the camera with a wobbling tree and a plane on the background and, duration ~ 20 s, rpm ∈ [5500, 6500]

### FRED 0 & 1
These samples are taken from the larger [FRED](https://miccunifi.github.io/FRED/) dataset which includes event data and normal video footage of drones flying on various conditions. Since the samples include annotated trajectories they can be helpful in building a solution for drone tracking. You can convert the .raw event data to .dat format using the conversion scripts in [Metavision SDK](https://docs.prophesee.ai/stable/installation/index.html).

## File Structure

After downloading, place the .dat files in this directory:

```
data/
├── README.md (this file)
├── fan.dat (or similar)
└── drone.dat (or similar)
```

## Using the Dataset

### Fan Rotation Detection
```bash
uv run scripts/fan_rotation_detector.py data/fan.dat
```

### Drone Tracking
```bash
uv run scripts/drone_tracker.py data/drone.dat
```

### Motion Analysis
```bash
uv run scripts/motion_analyzer.py data/fan.dat --export results.json
```

## Data Format

Event camera recordings are stored in `.dat` format (Prophesee Metavision format):
- Binary format with ASCII header
- Each event: (x, y, timestamp, polarity)
- Timestamps in microseconds
- Resolution: typically 1280x720

See main README.md for more details on the file format.
