import asyncio
import websockets
import json

async def send_message():
    uri = "ws://192.168.254.103:8001/ws"

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to the server.")

                while True:
                    text = input("Enter a letter or word to send (or '1' to quit): ").strip()

                    if text.lower() == '1':
                        print("Closing connection...")
                        return  # Exit function

                    if text.replace(" ", "").isalpha():  # Allow words and spaces
                        await websocket.send(text)
                        print(f"Sent: {text}")

                        response = await websocket.recv()

                        try:
                            data = json.loads(response)
                            word = data.get("letter", "Unknown response")
                            message = data.get("message", "")
                            
                            print(f"Server response: {word}")
                            if message:
                                print(f"Message: {message}")

                        except json.JSONDecodeError:
                            print(f"Received non-JSON response: {response}")

                    else:
                        print("Please enter only letters and spaces.")
        except websockets.exceptions.ConnectionClosed:
            print("Connection lost. Reconnecting in 3 seconds...")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"An error occurred: {e}")
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(send_message())