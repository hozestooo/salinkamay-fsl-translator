# SalinKamay: A Gesture Recognition Technology for Filipino Sign Language Translator

## Table of Contents
1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [System Architecture](#system-architecture)
4.  [Hardware Requirements](#hardware-requirements)
5.  [Software Requirements & Setup](#software-requirements--setup)
    * [Python Backend](#python-backend)
    * [Android Frontend](#android-frontend)
6.  [Usage](#usage)
    * [Data Collection (Dynamic Gestures)](#data-collection-dynamic-gestures)
    * [Model Training](#model-training)
    * [Running the Real-time Recognition System](#running-the-real-time-recognition-system)
7.  [Project Structure (Suggested)](#project-structure-suggested)
8.  [Future Work](#future-work)

## Introduction
SalinKamay is a real-time, portable gesture recognition system designed to translate Filipino Sign Language (FSL) into text and speech, aiming to bridge the communication gap between Deaf and Hard-of-Hearing (DHH) individuals and the hearing community in the Philippines. This project utilizes computer vision and machine learning deployed on accessible hardware (Raspberry Pi) with a companion Android application for user interaction. It integrates recognition for both static FSL gestures (like fingerspelling and numbers) and dynamic FSL signs/phrases.

## Features
* **Static FSL Recognition:** Recognizes the FSL alphabet (A-Z), numbers (0-9), and several control/common signs (e.g., "Okay," "Thank You," "ILY," mode switches).
    * Employs a TensorFlow Lite model with Multi-Head Attention and Dense layers.
    * Processes features from a single hand (with left-hand mirroring).
    * Uses debouncing for stable output.
* **Dynamic FSL Recognition:** Recognizes a set of common FSL signs and phrases involving motion (e.g., "Hello," "Good morning," "Maybe" - current vocabulary of ~15-16 dynamic gestures).
    * Employs a TensorFlow Lite LSTM (Long Short-Term Memory) model.
    * Processes 25-frame sequences using features from both hands (92 features/frame).
    * Uses a sliding window approach with confidence thresholding and transition allowance for real-time output.
* **Real-time Performance:** Designed to run on a Raspberry Pi Zero 2W.
* **Portability:** Utilizes a power bank and custom 3D-printed case.
* **Android Application Interface:**
    * Displays recognized text in a chat format.
    * Provides Text-to-Speech (TTS) output for recognized signs (supports Filipino and English).
    * Includes a "Translate Activity" to visualize letter-by-letter sign components with images.
* **Mode Switching:** Allows switching between dynamic recognition and static (fingerspelling/number) recognition modes.

## System Architecture
The system consists of two main parts:
1.  **Python Backend (on Raspberry Pi):**
    * Captures video feed using Raspberry Pi Camera Module V2.
    * Uses MediaPipe Hands to extract 21 key landmarks for up to two hands.
    * Performs feature engineering (normalized relative coordinates + fingertip distances -> 46 features/hand).
    * Based on the active mode (Static or Dynamic):
        * **Static Mode:** Processes features from the first detected hand (mirrors left hand), feeds to the appropriate static TFLite model (Letter/Sign or Number), and applies debouncing.
        * **Dynamic Mode:** Concatenates features from both hands (92 features/frame), forms a 25-frame sequence, scales it, and feeds it to the LSTM TFLite model.
    * Sends recognized text/data via WebSocket(s) to the Android application.
      
2.  **Android Frontend (Kotlin):**
    * Connects as a WebSocket client to the Python backend.
    * `ChatActivity`: Receives recognized text, displays it in a chat interface, and uses TTS.
    * `TranslateActivity`: Receives detailed letter-by-letter breakdowns with Base64 encoded images and displays them.
    * Provides UI for navigation, settings, etc.

## Hardware Requirements
* Raspberry Pi Zero 2W
* Raspberry Pi Camera Module V2
* 16GB (or larger) microSD Card with Raspberry Pi OS
* 10000mAh Power Bank (5V, min 2.5A-3A output)
* 3D Printed Case (optional, for protection and portability)
* Device to run Android application (e.g., Android smartphone)
* Local Wi-Fi Network (for WebSocket communication between RPi and Android device)

## Software Requirements & Setup

### Python Backend
* Python 3.x (e.g., 3.7, 3.9)
* **Key Libraries:** OpenCV, MediaPipe, TensorFlow, NumPy, scikit-learn, Joblib, WebSockets (`websockets` library for `StaticDynamic_withServer.py` acting as a client, and `fastapi`, `uvicorn`, `python-multipart` for `server.py` acting as a server - *clarify if both are needed or if `StaticDynamic_withServer.py` becomes the server*).
* **Setup:**
    1.  Clone the repository.
    2.  Set up a Python virtual environment (recommended).
    3.  Install dependencies: `pip install -r requirements.txt`
    4.  Place pre-trained `.tflite` models (dynamic, static letter, static number) and the `.pkl` scaler file in the appropriate `model/` subdirectories (or update paths in scripts).
    5.  Update hardcoded paths in scripts (e.g., `image_directory` in `server.py`, CSV paths, RTSP URL if used) or modify scripts to use command-line arguments/config files. **(This is a crucial step for shareability!)**

### Android Frontend
* Android Studio (latest stable version recommended).
* Android SDK.
* Kotlin enabled.
* **Setup:**
    1.  Open the Android project folder in Android Studio.
    2.  Let Gradle sync and build the project.
    3.  Update WebSocket IP addresses in `ChatActivity.kt` (via `WebSocketManager`), `TranslateActivity.kt`, and potentially `server.py` or `StaticDynamic_withServer.py` (if it acts as a client) to match your Raspberry Pi's IP address on the local network.

## Usage

### Data Collection (Dynamic Gestures)
* Run the `entry_dynamic_twoHand.py` script.
    * **Note:** This script currently has a hardcoded label and CSV output path. Modify it to allow dynamic label input (e.g., via console) and configurable output paths for usability.
    * Ensure the RTSP camera stream from the Raspberry Pi is accessible if `ip_address_rpicam` is used.

### Model Training
* **Static Models:** Use `keypoint_classification.ipynb` (or its `.py` equivalent).
    * Requires a `right_keypoint.csv` (and similarly for numbers) in the specified format (label, 46 features).
    * Update input CSV paths and model save paths as needed.
* **Dynamic Model:** Use `twoHands_training.ipynb` (or its `.py` equivalent).
    * Requires a CSV dataset (e.g., `dynamic_keypoint_two_hands.csv`) with format (label, 2300 features).
    * Update input CSV path and model/scaler save paths as needed.

### Running the Real-time Recognition System
1.  **Ensure Backend is Running:**
    * **If `StaticDynamic_withServer.py` is the main recognition engine AND also the WebSocket server:**
        * It needs to be modified to include the FastAPI/Uvicorn server startup code from `server.py` (the `if __name__ == "__main__":` block with `subprocess.Popen`).
        * Run `python StaticDynamic_withServer.py`.
    * **If `StaticDynamic_withServer.py` sends data TO `server.py` (which hosts the WS server):**
        * First, run `python server.py` (ensure Uvicorn is installed).
        * Then, run `python StaticDynamic_withServer.py` (ensure `ip_address_server` in its `send_letter` function points to where `server.py` is running).
2.  **Ensure Correct IP Addresses:** Verify all WebSocket URLs in the Python and Kotlin files point to the correct IP address of the machine running the WebSocket server(s) on your local network.
3.  **Run Android Application:** Build and run the app on an Android device connected to the same Wi-Fi network. Navigate to `ChatActivity` or `TranslateActivity`.

## Project Structure

Below is the suggested directory structure for the SalinKamay project:

```text
salinkamay/
├── .gitignore
├── README.md
├── python_backend/
│   ├── main_recognition_script.py  # Main script for real-time recognition & WebSocket output
│   ├── server.py                   # Optional: If running a separate FastAPI server for specific endpoints (e.g., image lookups for TranslateActivity)
│   ├── requirements.txt            # Python dependencies
│   ├── config.py                   
│   ├── data_collection/
│   │   └── entry_dynamic_twoHand.py
│   ├── training/
│   │   ├── static_model_training.ipynb   
│   │   └── dynamic_model_training.ipynb  
│   ├── models/
│   │   ├── static/
│   │   │   ├── right_keypoint_classifier.tflite
│   │   │   ├── right_digits_keypoint_classifier.tflite
│   │   │   ├── keypoint_classifier_label.csv
│   │   │   └── keypoint_digit_classifier_label.csv
│   │   └── dynamic/
│   │       ├── dynamic_classifier_two_hands.tflite
│   │       ├── scaler_two_hands.pkl
│   │       └── dynamic_classifier_label.csv
│   ├── utils/
│   │   └── hand_gesture_classifier.py
│   └── data/                       
│       ├── right_keypoint.csv
│       └── dynamic_keypoint_two_hands.csv
└── apk/
    └── SalinKamay.apk             
```

## Future Work

* **Expand Dynamic Gesture Vocabulary:**
    * Further increase the number of recognizable dynamic FSL signs and phrases.
    * New gestures can be added to the dataset relatively easily using the provided `entry_dynamic_twoHand.py` script for data collection.
    * The recommendation is to collect approximately 200 diverse video clips per new gesture to ensure robust model training.

* **Enhance Dynamic Model Performance:**
    * Continuously improve the dynamic model's accuracy and robustness.
    * Incorporate more varied training data, including additional signers and a wider range of environmental conditions.

* **Integrate Non-Manual Features:**
    * Incorporate the analysis of non-manual signals (such as facial expressions and body posture), which are crucial components of Filipino Sign Language.
    * This aims to enhance contextual understanding and overall translation accuracy for a richer FSL interpretation.
