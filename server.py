import asyncio
import base64
import os
import subprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import io
from PIL import Image

import csv

# --- Global Image Encoding Function ---
async def encode_image(image_path):
    """Read and encode an image to Base64"""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGBA')
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/png;base64,{base64_encoded}"
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

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
    , "pangit_ikaw.png":"PANGET MO", "sorry.png":"SORRY", "hello.png":"HELLO","maybe.png":"MAYBE"
    ,"nice_to_meet_you.png":"NICE TO MEET YOU"
}

with open('model/keypoint_classifier/dynamic_classifier_label.csv',
              encoding='utf-8-sig') as f:
        dynamic_classifier_labels = csv.reader(f)
        dynamic_classifier_labels = [
            row[0] for row in dynamic_classifier_labels
        ]

# Directory where images are stored
image_directory = r"C:\Users\harold\Downloads\sign\sign"

matches = []

# --- Server v1 ---
app_v1 = FastAPI()
active_connections_v1 = []

@app_v1.websocket("/ws")
async def websocket_endpoint_v1(websocket: WebSocket):
    await websocket.accept()
    active_connections_v1.append(websocket)

    try:
        while True:
            received_text = await websocket.receive_text()
            received_text = received_text.strip()

            if not received_text:
                continue  # Ignore empty messages

            print(f"Received: {received_text}")

            for phrase in dynamic_classifier_labels:
                index = received_text.find(phrase)
                if index != -1:
                    matches.append((index, phrase))
            
            if len(received_text) > 1:  # If it's Dynamic Gesture
                
                if received_text in dynamic_classifier_labels:
                    last_letter = received_text
                elif matches:
                    matches.sort()
                    last_letter = matches[-1][1]
                else:
                    last_letter = received_text[-1].upper()  # Get last letter

            else:
                last_letter = received_text.upper()  # Single letter

            # Find image for the last letter
            matching_image = next((img for img, l in image_data.items() if l == last_letter), None)

            if matching_image:
                image_path = os.path.join(image_directory, matching_image)

                if os.path.exists(image_path):
                    image_encoded = await encode_image(image_path)

                    response = {
                        "image": image_encoded,
                        "letter": received_text,  # Show the full word
                        "message": f"Sign language image for '{received_text}' (Last letter: {last_letter})"
                    }
                    for connection in active_connections_v1:
                        await connection.send_json(response)
                else:
                    await websocket.send_json({"error": f"Image for letter {last_letter} not found"})
            else:
                await websocket.send_json({"error": f"No image found for letter {last_letter}"})

    except WebSocketDisconnect:
        active_connections_v1.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections_v1:
            active_connections_v1.remove(websocket)
        await websocket.close()

# --- Server v2 ---
app_v2 = FastAPI()
active_connections_v2 = []
letter_to_image = {letter: img for img, letter in image_data.items()}

@app_v2.websocket("/ws")
async def websocket_endpoint_v2(websocket: WebSocket):
    await websocket.accept()
    active_connections_v2.append(websocket)

    try:
        while True:
            received_text = await websocket.receive_text()
            received_text = received_text.strip().lower()  # Convert to lowercase

            if not received_text:
                continue  # Ignore empty messages

            # Process each letter in the word
            letter_images = []
            
            for letter in received_text:
                letter_upper = letter.upper()
                
                if letter_upper in letter_to_image:
                    image_path = os.path.join(image_directory, letter_to_image[letter_upper])
                    
                    if os.path.exists(image_path):
                        image_encoded = await encode_image(image_path)
                        letter_images.append({
                            "letter": letter,
                            "image": image_encoded
                        })
                    else:
                        letter_images.append({
                            "letter": letter,
                            "error": f"Image file not found for letter {letter}"
                        })
                else:
                    letter_images.append({
                        "letter": letter,
                        "error": f"No image mapping for letter {letter}"
                    })
            
            # Send all letter images as a single response
            response = {
                "word": received_text,
                "letters": letter_images
            }
            
            await websocket.send_json(response)

    except WebSocketDisconnect:
        active_connections_v2.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections_v2:
            active_connections_v2.remove(websocket)
        await websocket.close()

# --- Start Both Servers ---
if __name__ == "__main__":
    # Run both servers in parallel
    process_v1 = subprocess.Popen(["uvicorn", "server:app_v1", "--host", "0.0.0.0", "--port", "8001"])
    process_v2 = subprocess.Popen(["uvicorn", "server:app_v2", "--host", "0.0.0.0", "--port", "8002"])

    # Keep the main process alive
    try:
        process_v1.wait()
        process_v2.wait()
    except KeyboardInterrupt:
        process_v1.terminate()
        process_v2.terminate()
