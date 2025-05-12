# Self-Driving Car Simulation

This project implements a self-driving car simulation using pure image processing techniques for lane detection, obstacle detection, and decision-making. The system processes video frames to identify lanes, detect objects, and make driving decisions based on the detected environment.

## Features

- **Lane Detection**: Detects left and right lanes using Hough Transform and adaptive smoothing.
- **Obstacle Detection**: Identifies objects in the region of interest (ROI) and classifies them as dangerous based on their proximity to the car.
- **Decision Making**: Implements logic to decide whether to move forward, stop, or steer left/right based on lane visibility and obstacle detection.
- **Debugging Tools**: Provides advanced debugging views for grayscale, edge detection, ROI masking, and detected lines.

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Imutils

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/waseemnabi08/self-driving-simulation
   cd <repository-folder>
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The `CONFIG` dictionary in `main.py` allows you to customize various parameters:

- **Edge Detection**:
  - `canny_low`, `canny_high`: Thresholds for Canny edge detection.
- **Hough Transform**:
  - `rho`, `theta`, `threshold`, `minLineLength`, `maxLineGap`: Parameters for line detection.
- **ROI Settings**:
  - `roi_height_ratio`, `roi_width_margin`: Define the region of interest for lane and object detection.
- **Object Detection**:
  - `min_object_area`, `min_object_width`, `min_object_height`: Filters for object size.
  - `danger_zone_threshold`: Defines the danger zone as a percentage of frame height.
- **General**:
  - `scale_factor`: Resizes the video frames for faster processing.
  - `debug_level`: Controls the level of debugging information displayed.

## Usage

1. Place your video file in the project directory.
2. Update the video file path in the `process_video` function call in `main.py`:
   ```python
   process_video("<your-video-file>.mp4")
   ```
3. Run the script:
   ```bash
   python main.py
   ```
4. Press `q` to exit the simulation.

## Debugging

Set `debug_level` in the `CONFIG` dictionary to:
- `0`: No debug views.
- `1`: Basic debug information.
- `2`: Advanced debug views, including grayscale, edge detection, and detected lines.

## Project Structure

- `main.py`: Main script containing the simulation logic.
- `requirements.txt`: List of required Python packages.

