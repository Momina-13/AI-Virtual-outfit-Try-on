# AI Virtual Outfit Try-On System

An AI-powered virtual try-on system that lets you see how clothes look on you in real time using your device's camera, running entirely in Google Colab.

## Features

- **Real-Time Camera Feed** – Access your device camera directly in Google Colab
- **Pose Detection** – MediaPipe-based body landmark detection (shoulders, hips, torso)
- **Virtual Clothing Overlay** – Intelligent garment placement aligned to your detected pose
- **Clothing Catalog** – Sample shirts and jackets included
- **Screenshot Capture** – Save and download virtual outfit images
- **Interactive UI** – ipywidgets-powered controls for clothing selection and camera management

## Project Structure

```
AI-Virtual-outfit-Try-on/
├── AI_Virtual_Outfit_Tryon.ipynb   # Main Google Colab notebook
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── clothing_samples/                # Sample clothing images
│   ├── shirt1.png
│   ├── shirt2.png
│   ├── jacket1.png
│   └── jacket2.png
└── utils/
    ├── __init__.py
    ├── pose_detector.py             # Pose detection utilities
    ├── clothing_overlay.py          # Clothing overlay functions
    └── camera_utils.py              # Camera handling for Colab
```

## Quick Start (Google Colab)

1. **Open the notebook** – Upload `AI_Virtual_Outfit_Tryon.ipynb` to Google Colab or open it directly from GitHub.

2. **Install dependencies** – Run the first cell which installs all required packages:
   ```python
   !pip install -r requirements.txt
   ```

3. **Allow camera access** – When prompted by your browser, allow Colab to access your webcam.

4. **Select clothing** – Use the dropdown menu to pick a garment from the catalog.

5. **Start camera** – Click **Start Camera** to begin the live try-on session.

6. **Take a screenshot** – Click **Capture Screenshot** to save a photo of your virtual outfit.

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Computer vision and frame processing |
| `mediapipe` | Real-time pose detection and body landmarks |
| `Pillow` | Image manipulation and clothing overlay |
| `numpy` | Numerical array operations |
| `ipywidgets` | Interactive UI components in Colab |
| `matplotlib` | Image display utilities |

Install all at once:
```bash
pip install -r requirements.txt
```

## How It Works

1. **Pose Detection** (`utils/pose_detector.py`)  
   MediaPipe Pose processes each camera frame to detect 33 body landmarks. Key points (shoulders, hips) are extracted to measure body dimensions.

2. **Clothing Overlay** (`utils/clothing_overlay.py`)  
   The clothing image is scaled and positioned based on shoulder width and torso length derived from the detected pose. Alpha-channel blending produces a realistic overlay.

3. **Camera Utilities** (`utils/camera_utils.py`)  
   Handles JavaScript-based webcam capture within Google Colab, encoding frames as base64 for Python processing.

4. **Colab Notebook** (`AI_Virtual_Outfit_Tryon.ipynb`)  
   Ties everything together with an interactive widget-based UI.

## Usage Tips

- Ensure good lighting for accurate pose detection.
- Stand ~1–2 metres from the camera so your full torso is visible.
- Keep a plain background for cleaner clothing overlays.
- Use the **confidence threshold slider** to balance detection sensitivity vs. stability.

## Troubleshooting

| Issue | Solution |
|-------|---------|
| Camera not starting | Allow browser camera permissions; re-run the camera cell |
| No pose detected | Improve lighting; make sure your torso is fully in frame |
| Clothing misaligned | Adjust the offset sliders in the UI or stand farther from the camera |
| Slow performance | Reduce frame resolution in `camera_utils.py` (default 640×480) |

## License

This project is for educational purposes demonstrating computer vision and machine learning in fashion technology.
