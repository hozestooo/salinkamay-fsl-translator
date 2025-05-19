import asyncio
import base64
import os
import subprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import io
from PIL import Image

# --- Server v1 ---

app_v1 = FastAPI()

# Dictionary of images and corresponding letters
image_data = {
    "a.png": "A", "b.png": "B", "c.png": "C", "d.png": "D",
    "e.png": "E", "f.png": "F", "g.png": "G", "h.png": "H",
    "i.png": "I", "j.png": "J", "k.png": "K", "l.png": "L",
    "m.png": "M", "n.png": "N", "o.png": "O", "p.png": "P",
    "q.png": "Q", "r.png": "R", "s.png": "S", "t.png": "T",
    "u.png": "U", "v.png": "V", "w.png": "W", "x.png": "X",
    "y.png": "Y", "z.png": "Z"
}

# Directory where images are stored (Update this path)
image_directory = r"C:\Users\User\Downloads\sign"

# Active WebSocket connections for v1
active_connections_v1 = []

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

            if len(received_text) > 1:  # If it's a word
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

# Reverse lookup dictionary for v2
letter_to_image = {letter: img for img, letter in image_data.items()}

# Active WebSocket connections for v2
active_connections_v2 = []

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


# Run both servers concurrently using subprocess

if __name__ == "__main__":
    # Update 'server' with the correct script name if it's different.
    subprocess.Popen(["uvicorn", "server:app_v1", "--host", "0.0.0.0", "--port", "8001"])
    subprocess.Popen(["uvicorn", "server:app_v2", "--host", "0.0.0.0", "--port", "8002"])

    # Keep the main process alive while the subprocesses are running
    while True:
        pass
