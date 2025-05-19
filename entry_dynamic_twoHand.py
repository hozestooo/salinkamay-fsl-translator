import time
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque 

import os
from PIL import Image
import string

import cv2 as cv
import numpy as np
import mediapipe as mp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=320)
    parser.add_argument("--height", help='cap height', type=int, default=240)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence', type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence', type=float, default=0.45)
    args = parser.parse_args()
    return args

# --- logging_csv() function - Saves 1840 features ---
def logging_csv(number, landmark_list):
    # Input CSV path here
    csv_path = r''
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
             header = ['label'] + [f'feature_{i}' for i in range(20 * 92)] # 20 frames * 92 features/frame
             writer.writerow(header)
        writer.writerow([number, *landmark_list]) # landmark_list should have 1840 elements
    print(f"Successfully saved data for label {number}")

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

    final_features = norm_coords # 42 normalized coords

    final_features.append(float(distance_ring_pinkie))
    final_features.append(float(distance_thumb_pinkie))
    final_features.append(float(distance_thumb_index))
    final_features.append(float(distance_index_middle))

    return final_features # Returns list of 46 features

def groupedProcessLandmark(image, hand_landmarks):
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

def main():
    args = get_args()
    # Input IP Address of video source (Raspberry Pi Camera)
    ip_address_rpicam = rf''

    rtsp_url = rf"rtsp://{ip_address_rpicam}:8554/MyStreamName"

    cap = cv.VideoCapture(rtsp_url)

    # Setup MediaPipe Hands for TWO hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2, 
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    max_frames_recorded = 25 # Number of frames where AT LEAST ONE hand is present
    num_features_per_hand = 46
    num_features_per_frame = num_features_per_hand * 2 # 92 features total per frame

    record = False
    countdown_active = False
    countdown_duration = 3 # Seconds
    countdown_start_time = 0 
    frame_counter = 0 
    sequence_data = [] 

    # HARD CODED LABEL HERE !!! ------------------------------------------------
    label_to_record = 13
    # HARD CODED LABEL HERE !!! ------------------------------------------------

    feedback_message = ""
    feedback_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        image = cv.flip(frame, 1)
        debug_image = copy.deepcopy(image)
        key = cv.waitKey(1) & 0xFF

        # --- Key Handling ---
        if key == ord('q'):
            break
        # Start countdown only if not already recording or counting down
        elif key == ord('k') and not record and not countdown_active:
            # Start countdown for the hardcoded label
            countdown_active = True
            countdown_start_time = time.time()
            feedback_message = "" # Clear any previous message
            print(f"Starting countdown for label {label_to_record}...")


        # --- Countdown Logic ---
        if countdown_active:
            elapsed_countdown = time.time() - countdown_start_time
            remaining_time = countdown_duration - elapsed_countdown

            if remaining_time <= 0:
                # Countdown finished, start recording
                countdown_active = False
                record = True
                frame_counter = 0 # Reset hand-present frame counter
                sequence_data = [] # Clear data for new sequence
                feedback_message = f"REC Label {label_to_record}"
                feedback_timer = time.time() + 1 # Show REC message briefly
                print(f"Recording gesture for label {label_to_record}...")
            else:
                # Display countdown timer
                countdown_text = f"Get Ready! Starting in {int(remaining_time) + 1}..."
                cv.putText(debug_image, countdown_text, (args.width // 4, args.height // 2),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
                # Keep processing hands during countdown to show user
                image_rgb_cd = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image_rgb_cd.flags.writeable = False
                hand_results_cd = hands.process(image_rgb_cd)
                image_rgb_cd.flags.writeable = True
                if hand_results_cd.multi_hand_landmarks:
                     for hand_landmarks in hand_results_cd.multi_hand_landmarks:
                         mp.solutions.drawing_utils.draw_landmarks(
                             debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- Recording Logic ---
        if record:
            # This block only runs *after* countdown is finished
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            hand_results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            # --- Record frame data ONLY if at least one hand is present ---
            if hand_results.multi_hand_landmarks:
                frame_counter += 1 # Increment counter only when hand(s) are found
                print(f"Hand(s) present, frame count: {frame_counter}") # Console feedback

                # Initialize features for the frame with padding (zeros)
                left_features = [0.0] * num_features_per_hand
                right_features = [0.0] * num_features_per_hand

                # Loop through detected hands (max 2)
                for hand_landmarks, handedness_info in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    # Draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Process landmarks for this specific hand
                    processed_features = groupedProcessLandmark(image, hand_landmarks) # 46 features

                    # Get handedness label ('Left' or 'Right')
                    hand_label = handedness_info.classification[0].label

                    # Fill the corresponding feature list
                    if hand_label == "Left":
                        left_features = processed_features
                    elif hand_label == "Right":
                        right_features = processed_features

                # Concatenate Left and Right features (always 92 features total)
                # Order: Left Hand Features (0-45), Right Hand Features (46-91)
                frame_features = left_features + right_features
                sequence_data.append(frame_features) # Append the 92 features for this valid frame

                # --- Display recording status (shows hand-present frame count) ---
                cv.putText(debug_image, f"REC Label {label_to_record} Frame: {frame_counter}/{max_frames_recorded}",
                           (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)

                # --- Check if required number of hand-present frames is collected ---
                if frame_counter == max_frames_recorded:
                    print(f"Collected {max_frames_recorded} hand-present frames for label {label_to_record}.")
                    # Flatten the sequence data for CSV saving
                    # Shape goes from (20, 92) -> (1840,)
                    features_to_save = np.array(sequence_data, dtype=np.float32).flatten().tolist()

                    # Save to CSV
                    logging_csv(label_to_record, features_to_save)

                    # Reset state
                    record = False # Stop recording
                    frame_counter = 0
                    sequence_data = []
                    # Keep label_to_record as is, or prompt again? For now, it stays 4 until 'k' is pressed again.
                    feedback_message = "Saved!"
                    feedback_timer = time.time() + 2 # Show message for 2 seconds
            # else: (No hands detected during recording)
                # Do nothing, wait for hands to appear to increment frame_counter

        # --- Display Feedback Message (if any) ---
        if feedback_message and time.time() < feedback_timer:
             color = (0, 255, 0) if feedback_message == "Saved!" else (0, 0, 255) # Green for Saved, Red for REC
             cv.putText(debug_image, feedback_message, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)
        elif feedback_message and time.time() >= feedback_timer:
             feedback_message = "" # Clear message

        # Show the final image for this iteration
        cv.imshow('Data Collection - Two Hands (Conditional Frame)', debug_image) # Changed window title
        
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
