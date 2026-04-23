# mini-dj

## Repository Structure

```text
dj-drew/
├── main.py                    # Entry point; sets up the app loop, connects gestures to DJ actions, and draws the UI.
|
├── hand_tracking/             # Module handling all computer vision and gesture recognition.
│   ├── __init__.py            
│   ├── classifier.py          # Builds/loads the PyTorch model, normalizes hand landmarks, and classifies the gestures.
│   └── tracker.py             # Interfaces with MediaPipe to extract hand landmarks and track pinch positions.
|
├── playback/                  # Module handling audio playback and UI rendering.
│   ├── __init__.py            
│   ├── selector.py            # Manages audio streams, stems, BPM sync, seeking, and the memory cue point.
│   └── ui.py                  # Defines the visual and interactive UI components (buttons, sliders, waveforms, decks).
|
├── tools/                     # Utility scripts for building and managing the gesture recognition pipeline.
│   ├── __init__.py            
│   ├── audit.py               # Identifies and removes "none" class data that visually conflicts with real gestures.
│   ├── collect.py             # Script to capture new hand landmark data from webcam and save to gesture_data.csv.
│   ├── test.py                # Standalone script to visually test gesture recognition accuracy without the DJ UI.
│   └── train.py               # Script to train the PyTorch gesture model from the CSV data.
|
├── models/                    # Binary and generated model artifacts.
│   ├── gesture_encoder.joblib # The label mappings for the trained PyTorch model.
│   ├── gesture_model.pt       # The trained PyTorch model weights.
│   └── hand_landmarker.task   # The base MediaPipe model used by hand_tracking/tracker.py.
|
├── data/                      # Training data storage.
│   └── gesture_data.csv       # The database of extracted hand landmarks collected using tools/collect.py.
|
├── songs/                     # Directory for music files.
│   └── ...                    # Stems and metadata (e.g. bass.mp3, bpm.txt)
|
├── requirements.txt           # Python dependencies for the project.
└── .gitignore                 # Specifies files for Git to ignore (like .venv, __pycache__, and models/data folders).
```
