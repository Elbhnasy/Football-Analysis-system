# Football Analysis System

![GitHub license](https://img.shields.io/github/license/Elbhnasy/Football-Analysis-system)
![GitHub issues](https://img.shields.io/github/issues/Elbhnasy/Football-Analysis-system)
![GitHub stars](https://img.shields.io/github/stars/Elbhnasy/Football-Analysis-system)

## Overview

The Football Analysis System is a comprehensive tool designed to track and analyze football players, referees, and the ball in video footage. This system leverages machine learning, computer vision, and deep learning techniques to provide detailed tracking information, which can be used for performance analysis, strategy development, and more.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
    - [Tracking Objects](#tracking-objects)
    - [Reading from Stub File](#reading-from-stub-file)
6. [Examples](#examples)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)
10. [Contact](#contact)

## Features

- **⚽ Object Detection and Tracking**: Utilizes [YOLO](https://github.com/ultralytics/yolov8), a state-of-the-art object detector, to detect players, referees, and footballs. Tracks these objects across frames using advanced tracking algorithms.
- **📊 Data Export**: Exports tracking data for further analysis.
- **📁 Stub File Support**: Optionally read tracking data from a stub file for faster processing.
- **🎨 Team Assignment**: Assigns players to teams based on the colors of their t-shirts using [KMeans](https://scikit-learn.org/stable/modules/clustering.html#k-means) for pixel segmentation and clustering.
- **🎥 Camera Movement Estimation**: Uses optical flow to measure camera movement between frames, enabling accurate measurement of player movement.
- **🔄 Perspective Transformation**: Represents the scene's depth and perspective, allowing measurement of player movement in meters rather than pixels.
- **🏃 Speed and Distance Calculation**: Calculates a player's speed and the distance covered.

## Dataset

The dataset used for this project includes 663 images annotated in YOLO v5 PyTorch format. It was exported via [Roboflow](https://api.roboflow.com/) on December 5, 2022. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### Dataset Structure

- **Images**: 663 images of football players, referees, and the ball.
- **Annotations**: YOLO v5 PyTorch format annotations.

Robowflow Football Dataset: [Roboflow Football Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)

## Prerequisites

- Python 3.9 or higher
- [pip](https://pip.pypa.io/en/stable/) (Python package installer)

## Installation

1. Clone the repository:
    ```bash
    git clone git@github.com:Elbhnasy/Football-Analysis-system.git
    cd Football-Analysis-system
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from [Roboflow](https://api.roboflow.com/) and place it in the `datasets/football-players-detection-1` directory.

## Usage

### Tracking Objects

- To track objects in video frames, use the `tracker.py` script. You can either provide a list of frames directly or read from a stub file.

```python
from trackers.tracker import Tracker

# Initialize the tracker
tracker = Tracker(model_path='path/to/yolo/model')

# List of frames (numpy arrays)
frames = [...]

# Get object tracks
tracks = tracker.get_object_tracks(frames, read_from_stub=False, stub_path=None)

# Process the tracks as needed
```

### Command-Line Usage
- You can also use the command line to run the tracker:

```bash
python trackers/tracker.py --video_path path/to/video.mp4 --output_path path/to/output
```

### Reading from Stub File
- If you have a stub file with precomputed tracks, you can read from it to save time.

```python
tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='path/to/stub/file')
```
## Examples
- Here are some examples of the tracking system in action:

- ![download](https://github.com/user-attachments/assets/4503259c-7bde-440a-b13d-6ca3c37f017f)
- ![download (3)](https://github.com/user-attachments/assets/0825bb41-75f5-4595-9d8e-613c20317ece)
- ![download (7)](https://github.com/user-attachments/assets/9c45afc7-244c-4b21-a662-293c971eb2ab)
- ![download (5)](https://github.com/user-attachments/assets/22917152-ce0f-43d5-8109-62f566a39a41)
- ![download (6)](https://github.com/user-attachments/assets/3335ad43-bc2a-4132-9023-b89749b80cf6)


- https://github.com/user-attachments/assets/ed2b8493-faed-4464-a09d-b5dbf208b99b





## Contributing

We welcome contributions to improve the **Football Analysis System**. If you'd like to contribute, please follow these steps:

1. **Fork the repository**

   Click the "Fork" button on the top-right corner of this repository's page to create a copy under your GitHub account.

2. **Clone your fork**

   ```bash
   git clone https://github.com/your-username/Football-Analysis-system.git
   cd Football-Analysis-system

## License

- This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

## Credits

- **[Roboflow](https://roboflow.com/)**: To provide the tools to annotate and export the dataset used in this project.
- **[YOLO v5](https://github.com/ultralytics/yolov5)**: For the object detection framework that powers the tracking and detection features in this system.

## Contat

Please open an issue in this repository or contact me directly at **khaledtelbahnasy@gmail.com**.

