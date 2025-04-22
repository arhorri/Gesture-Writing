# ArUco Marker Detection with 3D Overlays

This Python application detects ArUco markers in a video stream, defines a region of interest (ROI) based on these markers, runs object detection within the ROI using YOLO, and overlays 3D models on detected objects.

## Features

- ArUco marker detection using OpenCV
- Marker pose estimation
- Region of Interest (ROI) definition based on markers
- YOLO object detection within the ROI
- 3D model overlays using OpenCV's perspective projection
- Support for custom 3D models (OBJ format)
- Camera calibration for improved accuracy

## Requirements

- Python 3.7+
- Webcam or video file
- Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the YOLO model (happens automatically on first run)

## Usage

### Basic Usage

Run the application with the default settings:
```
python marker_detection_app.py
```

### Advanced Usage

```
python marker_detection_app.py --camera 0 --marker-size 0.05 --calibration camera_calibration.json --model path/to/model.obj
```

Options:
- `--camera`: Camera index (default: 0)
- `--video`: Path to video file (instead of using camera)
- `--marker-size`: Physical size of the marker in meters (default: 0.05)
- `--calibration`: Path to camera calibration file (default: camera_calibration.json)
- `--model`: Path to custom 3D model file (OBJ format)

### Generating ArUco Markers

Generate printable markers:
```
python generate_markers.py
```

This will create several ArUco markers in the `markers` directory. Print these markers and place them in your scene.

### Camera Calibration

For better accuracy, calibrate your camera:

1. Capture calibration images:
   ```
   python camera_calibration.py --capture
   ```
   Hold a chessboard pattern (default is 9x6 internal corners) in front of the camera and press 'c' to capture images.

2. Run calibration:
   ```
   python camera_calibration.py --calibrate
   ```

3. To do both steps at once:
   ```
   python camera_calibration.py --capture --calibrate
   ```

### Custom 3D Models

You can use custom 3D models in OBJ format:

1. Preview a 3D model:
   ```
   python load_obj_model.py path/to/model.obj --preview
   ```

2. Run the application with a custom model:
   ```
   python marker_detection_app.py --model path/to/model.obj
   ```

## Controls

- Press 'q' or ESC to quit the application
- Press 'f' to toggle the FPS display

## How It Works

1. The application detects ArUco markers in the video stream
2. It defines a region of interest (ROI) based on the detected markers
3. The ROI is warped to get a top-down view
4. YOLO object detection is run on the warped ROI
5. Detected objects are mapped back to the original frame
6. 3D models are overlaid at the detected object positions

## Customization

- Change the camera source (webcam index or video file path) in the constructor
- Use different ArUco dictionaries for different marker types
- Specify the marker size in meters for accurate pose estimation
- Swap the YOLO model for different object detection capabilities
- Use custom 3D models for more realistic overlays

# Finger Detection Application

This repository contains different implementations of finger detection applications using computer vision techniques.

## Requirements

The basic requirements for running the applications are:

```
opencv-contrib-python>=4.5.0
numpy>=1.19.0
```

For the advanced versions (if you have a compatible Python version), you'll also need:

```
mediapipe>=0.8.10
```

## Installation

1. Install the required dependencies:

```bash
pip install opencv-contrib-python numpy
```

## Applications

### 1. Simple Finger Detection (OpenCV only)

This application uses basic computer vision techniques with OpenCV to detect fingers. It works by detecting skin color and using contour analysis to count fingers.

To run:

```bash
python simple_finger_detection.py
```

Options:
- `--camera`: Camera index (default: 0)
- `--video`: Path to a video file instead of using the camera
- `--threshold`: Threshold for skin color detection (default: 25)

Controls:
- Press 'q' or ESC to quit
- Press 'f' to toggle FPS display
- Press 'm' to change the detection mode (Normal, Skin Detection, Contours)
- Press '+' or '-' to adjust the threshold

### 2. Advanced Finger Detection (Requires MediaPipe)

This application uses MediaPipe Hands for more accurate hand and finger tracking, with gesture recognition and special effects.

To run (requires a Python version compatible with MediaPipe, usually Python 3.7-3.10):

```bash
python advanced_finger_detection.py --effects
```

Options:
- `--camera`: Camera index (default: 0)
- `--video`: Path to a video file instead of using the camera
- `--max-hands`: Maximum number of hands to detect (default: 2)
- `--min-detection-confidence`: Minimum detection confidence (default: 0.5)
- `--min-tracking-confidence`: Minimum tracking confidence (default: 0.5)
- `--effects`: Enable special visual effects

Controls:
- Press 'q' or ESC to quit
- Press 'f' to toggle FPS display
- Press 'l' to toggle landmarks display
- Press 'e' to toggle special effects
- Press 'g' to toggle gesture recognition display
- Press 'm' to change the visual processing mode

## Gestures Recognized (Advanced Version)

- Fist: All fingers closed
- Open Hand: All fingers extended
- Pointing: Only index finger extended
- Victory: Index and middle fingers extended in a V shape
- Thumbs Up: Only thumb extended
- Pinch: Thumb and index finger close together

## Troubleshooting

If you encounter issues with MediaPipe installation, you may need to use a compatible Python version (typically 3.7-3.10). The simple version works with any Python version that supports OpenCV.

For Windows users, make sure you have the Visual C++ redistributable installed.

# Ichimoku Cloud Trading Bot

A Telegram bot that monitors cryptocurrency trading pairs using the Ichimoku Cloud indicator and sends alerts when trading signals are detected.

## Features

- Monitor multiple trading pairs (BTC/USDT, ETH/USDT, SOL/USDT, PEPE/USDT)
- Support for multiple timeframes (1m, 5m, 15m, 1h)
- Real-time Ichimoku Cloud signal detection
- Interactive setup process with inline keyboards
- Manual chart plotting functionality
- Automatic signal notifications with charts
- Clean and modern UI

## Prerequisites

- Python 3.8 or higher
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))
- Binance API credentials (optional, for higher rate limits)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ichimoku-trading-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
cp .env.example .env
```

5. Edit the `.env` file and add your credentials:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
```

## Usage

1. Start the bot:
```bash
python main.py
```

2. Open Telegram and start a chat with your bot

3. Send the `/start` command to see the main menu

4. Follow the interactive setup process to:
   - Set up trading pair alarms
   - Cancel existing alarms
   - Plot charts manually

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t ichimoku-bot .
```

2. Run the container:
```bash
docker run -d --name ichimoku-bot --env-file .env ichimoku-bot
```

## Project Structure

- `main.py` - Entry point
- `bot.py` - Telegram bot implementation
- `data_fetcher.py` - CCXT integration for fetching OHLCV data
- `ichimoku.py` - Ichimoku Cloud calculation logic
- `chart_plotter.py` - Chart generation using mplfinance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 