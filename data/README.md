# Event Camera Dataset

This directory contains event camera recordings (.dat files) for the hackathon challenge.

## Download Dataset

Download the dataset from the Google Drive link:
https://drive.google.com/drive/folders/18ORzE9_aHABYqOHzVdL0GANk_eIMaSuE?usp=sharing

The dataset includes:
- **Fan recordings**: For rotation detection and counting
- **Drone recordings**: For object tracking and speed estimation

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
