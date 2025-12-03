from furhat_realtime_api import FurhatClient
import time

host = "192.168.0.191"

def main():
    furhat = FurhatClient(host)
    furhat.connect()
    print(f"Connected to Furhat at {host}")
    
    try:
        furhat.request_attend_user()
        time.sleep(0.5)
        
        furhat.request_gesture_start(name="Smile", wait=True)
        time.sleep(0.5)
        
        furhat.request_speak_text(text="Hello, basic connection test.", wait=True)
        time.sleep(1)
        
    finally:
        furhat.disconnect()
        print("Disconnected from Furhat")

if __name__ == "__main__":
    main()
