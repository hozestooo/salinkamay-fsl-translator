import csv
import copy
import argparse
import itertools
import numpy as np
import cv2 as cv
import mediapipe as mp

from hand_gesture_classifier import HandGestureClassifier
from rapidfuzz import process

import asyncio
import base64
import os
import subprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from PIL import Image
import pandas as pd

import asyncio

import time
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import os

import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import websockets
import json

app_v1 = FastAPI()

# --- Shared Image Data ---
image_data = {
    # Alphabet
    "A.png": "A", "B.png": "B", "C.png": "C", "D.png": "D",
    "E.png": "E", "F.png": "F", "G.png": "G", "H.png": "H",
    "I.png": "I", "J.png": "J", "K.png": "K", "L.png": "L",
    "M.png": "M", "N.png": "N", "O.png": "O", "P.png": "P",
    "Q.png": "Q", "R.png": "R", "S.png": "S", "T.png": "T",
    "U.png": "U", "V.png": "V", "W.png": "W", "X.png": "X",
    "Y.png": "Y", "Z.png": "Z",

    # Numbers
    "1.png": "1", "2.png": "2", "3.png": "3", "4.png": "4",
    "5.png": "5", "6.png": "6", "7.png": "7", "8.png": "8",
    "9.png": "9", "0.png": "0",

    # Special ACRONYMS
    "ty.png": "TY", "ily.png": "ILY", "enye.png":"Ñ",    
    
    # Symbols
    "QM.png":"?", "EP.png":"!",

    # Modes
    "Number.png":"Number", "Letter.png":"Letter",

    # Symbols
    "Deleted.png":"d",

    # Dynamic Gestures
    "beautiful.png":"ANG GANDA MO", "deaf.png":"DEAF", "fingerspell.png":"FINGERSPELL", 
    "good_morning.png":"GOOD MORNING","good_afternoon.png":"GOOD AFTERNOON", 
    "good_evening.png":"GOOD EVENING", "kamusta_ka.png":"KAMUSTA KA NA?", 
    "my_name_is.png":"MY NAME IS", "no.png":"NO", "yes.png":"YES"
    , "pangit_ikaw.png":"PANGET MO", "sorry.png":"SORRY", "hello.png":"HELLO","maybe.png":"MAYBE",
    "nice_to_meet_you.png":"NICE TO MEET YOU"
}

image_directory = r"C:\Users\harold\Downloads\sign\sign"

active_connections_v1 = []

async def send_letter(label):
    # IP Address of device running server.py
    ip_address_server = ""
    uri = rf"ws://{ip_address_server}:8001/ws"

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to the server")
            
            if label.strip():
                await websocket.send(label) 
                print(f"Sent: {label}")  

                response = await websocket.recv()  # Wait for server response

                try:
                    data = json.loads(response)  # Parse JSON response
                    word = data.get("letter", "Unknown response")
                    message = data.get("message", "")

                    return True  
                
                except json.JSONDecodeError:
                    print(f"Received non-JSON response: {response}")
                    return False  # Indicate failure
    except websockets.exceptions.ConnectionClosed:
        print("Connection lost. Reconnecting in 3 seconds...")
        await asyncio.sleep(3)
    except Exception as e:
        print(f"An error occurred: {e}")
        await asyncio.sleep(3)

    return False

def send_message(message):
    success = asyncio.run(send_letter(message))
    if success: print("Sent sucessfully")
    else: print("Failed to send message")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    
    parser.add_argument("--fps", type=int, default=15, help="Set frames per second (default is 24)")

    args = parser.parse_args()
    return args

def getLabels():
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    
    with open('model/keypoint_classifier/keypoint_digit_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_digits_classifier_labels = csv.reader(f)
        keypoint_digits_classifier_labels = [
            row[0] for row in keypoint_digits_classifier_labels
        ]

    with open('model/keypoint_classifier/dynamic_classifier_label.csv',
              encoding='utf-8-sig') as f:
        dynamic_classifier_labels = csv.reader(f)
        dynamic_classifier_labels = [
            row[0] for row in dynamic_classifier_labels
        ]

    return keypoint_classifier_labels, keypoint_digits_classifier_labels, dynamic_classifier_labels

def dynamic_inference(input_data, interpreter, input_details, output_details):
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])
        confidence = np.max(output_data[0])

    except Exception as e:
        print(f"Error during inference: {e}")
    return predicted_class, confidence

def dynamic_model_initialize(dynamic_model_path):

    try:
        interpreter = tf.lite.Interpreter(model_path=dynamic_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Two-hand TFLite model loaded successfully.")
        print("Expected TFLite Input Shape:", input_details[0]['shape'])
        expected_shape = [1, 20, 92]
        if list(input_details[0]['shape']) != expected_shape:
             print(f"WARNING: Model input shape {input_details[0]['shape']} does not match expected {expected_shape}")

    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return
    
    return interpreter, input_details, output_details

def scaler_initialize(scaler_path):

    try:
        scaler = joblib.load(scaler_path)
        print("Two-hand scaler loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {scaler_path}")
        return
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return
    
    return scaler

def main():
    args = get_args()
    
    # Input directory path of base folder
    base_dir = r''
    
    # IP Address of video source 
    ip_address_rpicam = r''

    base_model_dir = base_dir
    dynamic_model_path = os.path.join(base_model_dir, 'dynamic_classifier_two_hands.tflite')
    scaler_path = os.path.join(base_model_dir, 'scaler_two_hands.pkl')

    scaler = scaler_initialize(scaler_path)
    
    right_model_path = "model/keypoint_classifier/model/right_keypoint_classifier.tflite"
    number_model_path = "model/keypoint_classifier/model/right_digits_keypoint_classifier.tflite"

    classifier = HandGestureClassifier(right_model_path, number_model_path)

    interpreter, input_details, output_details = dynamic_model_initialize(dynamic_model_path)

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    rtsp_url = rf"rtsp://{ip_address_rpicam}:8554/MyStreamName"
    cap = cv.VideoCapture(rtsp_url)

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier_labels, keypoint_digits_classifier_labels, dynamic_classifier_labels = getLabels()
        
    predicted_class = ""
    confidence = ""

    collected_text = ""
    previous_label = ""

    space_added = False

    MAX_FRAMES_BEFORE_CLEAR_COLLECTED = 1200 # Clear collected every nth frame
    MAX_FRAMES_BEFORE_SPACE = 18
    # For class prediction
    DETECTION_THRESHOLD_LETTERS = 5
    DETECTION_THRESHOLD_NUMBERS = 15

    model = "Dynamic"
    default_fingerspelling = "Letter"

    TRANSITION_FRAMES = 6 # Allowance frame to transition between dynamic
    transitionState = True
    NUM_FRAMES = 25
    FEATURES_PER_HAND = 46
    NUM_FEATURES_PER_FRAME = FEATURES_PER_HAND * 2 # 92
    # Use a deque, clearing it resets the history
    window = deque(maxlen=NUM_FRAMES)
    frame_count = 0
    pauseDynamic = 1
    static_hand_landmarks = 0
    label_form = ""
    occurences = 0
    SignValid = False
    previous_class = ""
    considered_class = ""
    staticTransitionState = True

    while True:
        ret, frame = cap.read()
        if not ret: break
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'): break

        image = cv.flip(frame, 1)
        debug_image = copy.deepcopy(image)

        # Process image with MediaPipe
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        hand_results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        left_features = [0.0] * FEATURES_PER_HAND
        right_features = [0.0] * FEATURES_PER_HAND

        if hand_results.multi_hand_landmarks:
            pauseDynamic = 1
            frame_count+=1
            detected_hands_count = len(hand_results.multi_hand_landmarks)

            # Transition Frame for Dynamic Gestures
            if transitionState and TRANSITION_FRAMES > 0 and model == "Dynamic":
                print("TRANSITIONING...")
                TRANSITION_FRAMES-=1
            elif TRANSITION_FRAMES == 0 and model == "Dynamic":
                print("TRANSITION DONE!")
                TRANSITION_FRAMES = 6
                transitionState = False

            # Transition Frame for Static Gestures
            # Only applied when changing modes
            if staticTransitionState and TRANSITION_FRAMES > 0 and model != "Dynamic":
                print("STATIC Transitioning...")
                TRANSITION_FRAMES-=1
            elif TRANSITION_FRAMES == 0 and model != "Dynamic":
                print("STATIC TRANSITION DONE!")
                TRANSITION_FRAMES = 6
                staticTransitionState = False

            for index in range(detected_hands_count):
                hand_landmarks = hand_results.multi_hand_landmarks[index]
                handedness_list = hand_results.multi_handedness[index]

                hand_label = handedness_list.classification[0].label

                mp.solutions.drawing_utils.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                processed_features = groupedProcessLandmark(image, hand_landmarks, hand_label, model)

                if index == 0 and model != "Dynamic":
                    static_hand_landmarks = processed_features

                if hand_label == "Left": left_features = processed_features
                elif hand_label == "Right": right_features = processed_features
            
            if model == "Dynamic" and transitionState is False:
                input_landmarks = left_features + right_features
                window.append(input_landmarks)

                if len(window) == NUM_FRAMES:
                    window_data = np.array(window, dtype=np.float32)
                    window_flat = window_data.reshape(1, -1) 
                    scaled_data_flat = scaler.transform(window_flat)
                    input_data = scaled_data_flat.reshape(1, NUM_FRAMES, NUM_FEATURES_PER_FRAME) 
                    input_data = input_data.astype(np.float32)
                    predicted_class, confidence = dynamic_inference(input_data, interpreter, input_details, output_details)

                    print(f"Predicted Class: {predicted_class} || Confidence: {confidence}")
                    
                    if predicted_class == 11 and confidence > .97:
                        print(f"Fingerspell activated: Default: {default_fingerspelling}")    
                        window.clear()
                        model = default_fingerspelling
                        space_added = True
                        
                    elif confidence > .97:
                        label_form = dynamic_classifier_labels[predicted_class]
                        collected_text+= label_form + " "
                        send_message(label_form)
                        print(f"Collected: {collected_text}")

                        if label_form == "MY NAME IS":
                            model = "Letter"
                            staticTransitionState = True
                            space_added = True
                            
                        window.clear()
                        transitionState = True

                    TRANSITION_FRAMES = 6
                    
                else: 
                    print(f"Collecting... ({len(window)}/{NUM_FRAMES})")
            
            elif detected_hands_count == 1 and static_hand_landmarks and model != "Dynamic" and staticTransitionState is False:
                
                # Fingerspelling
                predicted_class, confidence = classifier.run_inference(static_hand_landmarks, model)
                
                if confidence > .85:
                    
                    print(f"Previous_class: {previous_class} || current: {predicted_class} || Iteration: {occurences}")
                    
                    if previous_class != predicted_class:
                        # If it's a new detected class, consider it as previous
                        previous_class = predicted_class
                        occurences = 0
                    elif previous_class == predicted_class:
                        # If it's still the same value, increment occurence
                        occurences+=1
                    
                    print(f"Occurences: {occurences} || Model: {model}")

                    if occurences == DETECTION_THRESHOLD_LETTERS and model == "Letter":
                        # If it reached threshold, use as considered class
                        occurences = 0
                        considered_class = predicted_class
                        previous_class = 0
                        SignValid = True
                    elif occurences == DETECTION_THRESHOLD_NUMBERS and model == "Number":
                        occurences = 0
                        considered_class = predicted_class
                        previous_class = 0
                        SignValid = True

                    if SignValid:
                        # Get the label
                        if model == "Letter":
                            label_form = keypoint_classifier_labels[considered_class]
                        elif model == "Number":
                            # Excluding 31 as it's from model class
                            label_form = keypoint_digits_classifier_labels[considered_class]
                        
                        # Check if static model switch called
                        if model == "Letter" and label_form == "Number":
                            model = "Number"
                            staticTransitionState = True
                            print("Number Mode Activated")

                            if space_added == False:
                                collected_text+=" "
                                space_added = True
                            
                            label_form = ""
                        
                        elif model == "Number" and label_form == "Letter":
                            model = "Letter"
                            staticTransitionState = True
                            print("Letter Mode Activated")

                            if space_added == False:
                                collected_text+= " "
                                space_added = True
                            
                            label_form = ""
                        
                        # End of checking if model switch called

                        if label_form == "Not okay" and collected_text:
                            print(f"Deleted Text: {collected_text}")
                            label_form = ""
                            collected_text = ""
                        elif label_form == "Okay":
                            print(f"Send Text: {collected_text}")

                            send_message(collected_text)

                            label_form = ""
                            if collected_text == "DYNAMIC": 
                                model = "Dynamic"
                                transitionState = True

                                if space_added == False:
                                    collected_text += " " 
                                    space_added = True
                                
                            collected_text = ""
                            
                        elif label_form == "Thank you" and label_form != previous_label:
                            collected_text = "THANK YOU"
                        elif label_form == "ILY" and label_form != previous_label:
                            collected_text = "I LOVE YOU"
                        else:
                            if label_form and label_form !="Unknown":
                                if model == "Letter" and label_form != previous_label:
                                    # Add to letter collection
                                    send_message(label_form)
                                    collected_text += label_form
                                
                                elif model == "Number":
                                    send_message(label_form)
                                    collected_text += label_form
                        
                        # Records the previous label
                        previous_label = label_form

                    # Records the previous class
                    previous_class = predicted_class
                    
                    # Return to False
                    SignValid = False
                    
                print(f"Collected text: {collected_text}")

        else:
            if model == "Dynamic":

                if window and pauseDynamic % 20 == 0:
                    window.clear()

                if collected_text or window: pauseDynamic+=1

                if pauseDynamic == 80:
                    print(f"Send collected Text: {collected_text}")
                    send_message(collected_text)
                    collected_text = ""
                    pauseDynamic = 1
            
            else: pauseDynamic = 1
            
            if collected_text and frame_count >= MAX_FRAMES_BEFORE_CLEAR_COLLECTED:
                # Clear the considered classes
                print("Clear Collected")
                frame_count = 1
                collected_text = ""
                space_added = False
            elif frame_count % MAX_FRAMES_BEFORE_SPACE == 0 and collected_text and space_added is False and model !="Dynamic":
                print("Space simulation")
                collected_text+=" "
                space_added = True
            else:
                frame_count +=1

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()
        
def pre_process_landmark(landmark_list):
    base_x, base_y = 0, 0
    rel_coords = []
    for index, landmark_point in enumerate(landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        rel_x = landmark_point[0] - base_x
        rel_y = landmark_point[1] - base_y
        rel_coords.append(rel_x)
        rel_coords.append(rel_y)

    if not rel_coords: return [0.0] * 46
    max_value = max(map(abs, rel_coords))
    if max_value == 0: max_value = 1
    norm_coords = [n / max_value for n in rel_coords]

    try:
        thumb_landmark_x, thumb_landmark_y = norm_coords[8], norm_coords[9]
        index_landmark_x, index_landmark_y = norm_coords[16], norm_coords[17]
        middle_landmark_x, middle_landmark_y = norm_coords[24], norm_coords[25]
        ring_landmark_x, ring_landmark_y = norm_coords[32], norm_coords[33]
        pinkie_landmark_x, pinkie_landmark_y = norm_coords[40], norm_coords[41]

        distance_thumb_index = np.sqrt((index_landmark_x - thumb_landmark_x)**2 + (index_landmark_y - thumb_landmark_y)**2)
        distance_index_middle = np.sqrt((middle_landmark_x - index_landmark_x)**2 + (middle_landmark_y - index_landmark_y)**2)
        distance_ring_pinkie = np.sqrt((pinkie_landmark_x - ring_landmark_x)**2 + (pinkie_landmark_y - ring_landmark_y)**2)
        distance_thumb_pinkie = np.sqrt((pinkie_landmark_x - thumb_landmark_x)**2 + (pinkie_landmark_y - thumb_landmark_y)**2)
    except IndexError:
        print(f"Error: IndexError during distance calculation. norm_coords length: {len(norm_coords)}")
        return [0.0] * 46

    final_features = norm_coords

    final_features.append(float(distance_ring_pinkie))
    final_features.append(float(distance_thumb_pinkie))
    final_features.append(float(distance_thumb_index))
    final_features.append(float(distance_index_middle))

    return final_features 

def groupedProcessLandmark(image, hand_landmarks, hand_label, model):

    if model != "Dynamic" and hand_label == "Left":
        mirrored_landmarks = copy.deepcopy(hand_landmarks)

        for landmark in mirrored_landmarks.landmark:
            landmark.x = 1.0 - landmark.x
        
        landmark_list = calc_landmark_list(image, mirrored_landmarks)
    else:
        landmark_list = calc_landmark_list(image, hand_landmarks)

    pre_process_landmark_list = pre_process_landmark(landmark_list)
    return pre_process_landmark_list

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

if __name__ == '__main__':
    main()