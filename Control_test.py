# simple_controller.py

import time
import threading
# Import the class from your existing file
from Robo_Handtracking import RealTimeARClient

# --- Configuration ---
SERVER_URL = "https://balanced-vaguely-mastodon.ngrok-free.app/inference/"
DROIDCAM_URL = "http://192.168.133.7:4747/video"

def main():
    """
    Main function to run the interactive controller.
    """
    # 1. Create an instance of your AR client
    ar_client = RealTimeARClient(SERVER_URL, DROIDCAM_URL, initial_prompt="Waiting for command...")

    # 2. Run the client's main loop in a background thread.
    #    This is crucial because client.run() is a blocking call.
    client_thread = threading.Thread(target=ar_client.run, daemon=True)
    client_thread.start()
    print("[Controller] AR Client has started in the background.")
    # Give the client a moment to initialize the camera
    time.sleep(5) 

    # 3. This is your main control loop.
    #    It allows you to send commands to the running AR client.
    while True:
        print("\n--- Controller Menu ---")
        print(" 'p' - Set a new prompt")
        print(" 's' - Sample the current view (trigger detection)")
        print(" 'q' - Quit the application")
        choice = input("Enter your command: ").lower()

        if choice == 'p':
            new_prompt = input("  Enter the new prompt: ")
            ar_client.set_prompt(new_prompt)
        
        elif choice == 's':
            print("  Telling the client to sample the view...")
            ar_client.trigger_detection()

        elif choice == 'q':
            print("  Sending stop signal to the client...")
            ar_client.stop()
            break
        
        else:
            print("  Invalid command. Please try again.")
    
    # Wait for the client thread to finish cleanly
    client_thread.join(timeout=5)
    print("[Controller] Application has exited.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Controller] Manual shutdown requested.")

