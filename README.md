# Advanced Finger Detection & Gesture Recognition with MediaPipe

This project demonstrates an **Advanced Finger Detection and Gesture Recognition** system using **MediaPipe** and **OpenCV**. The application tracks hand gestures in real-time via a webcam or video file, allowing users to interact with a virtual canvas through gesture-based controls. Special effects such as particles and beams are generated based on recognized gestures.

## Features
- **Hand and Finger Tracking**: Detect and track individual fingers with precision.
- **Gesture Recognition**: Recognizes gestures including:
  - "Fist"
  - "Open Hand"
  - "Pointing"
  - "Victory"
  - "Thumbs Up"
  - "Pinch"
- **Interactive Drawing**: Use pinch gestures to draw lines and create virtual artwork.
- **Special Effects**: Particle effects and beams triggered by specific gestures.
- **Customizable Controls**: Easily toggle FPS, landmarks, gesture recognition, drawing settings, and effects.

## Technologies Used
- **Python**: Programming language used for this project.
- **MediaPipe**: For hand and finger tracking, and gesture recognition.
- **OpenCV**: For video capture, real-time processing, and visual effects.
- **NumPy**: For numerical operations and array handling.

## Installation

### Prerequisites
Ensure Python 3.x is installed on your system. Then, install the necessary dependencies:

```bash
pip install mediapipe opencv-python numpy
