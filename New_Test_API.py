# test_client.py

import requests
import os
from Resize import process_and_resize_image

# --- Configuration ---
# Replace this with the public URL provided by ngrok when you run the server
BASE_URL = "https://balanced-vaguely-mastodon.ngrok-free.app/" 
VERIFY_URL = f"{BASE_URL}verify"
PROMPT_URL = f"{BASE_URL}prompt"

# --- User Inputs ---
# Make sure this image file exists in the same directory as the script,
# or provide a full path to it.
IMAGE_PATH = process_and_resize_image("./assets/demo/Pemanas.png", 480) 
OBJECT_TO_VERIFY = input("Enter the object to verify in the image (e.g., 'a red heater'): ")

# --- Check if image exists ---
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: Image file not found at '{IMAGE_PATH}'")
    exit()

# This variable will store the ID we get from the server
verified_image_id = None

# --- Step 1: Call the /verify endpoint ---
print("\n" + "="*25)
print("STEP 1: VERIFYING IMAGE")
print("="*25)

try:
    # Prepare the data and file for the multipart/form-data request
    verify_data = {"object_id": OBJECT_TO_VERIFY}
    files = {"image": (os.path.basename(IMAGE_PATH), open(IMAGE_PATH, "rb"))}

    print(f"Sending '{OBJECT_TO_VERIFY}' and '{IMAGE_PATH}' to {VERIFY_URL}...")
    
    # Make the POST request
    response = requests.post(VERIFY_URL, data=verify_data, files=files)
    
    # Raise an exception for bad status codes (4xx or 5xx)
    response.raise_for_status()

    # If successful, get the image_id from the JSON response
    response_json = response.json()
    verified_image_id = response_json.get("image_id")
    
    if verified_image_id:
        print("✅ Verification successful!")
        print(f"   Received Image ID: {verified_image_id}")
    else:
        print("❌ Verification succeeded, but no image_id was returned.")

except requests.exceptions.HTTPError as err:
    print(f"❌ Verification Failed! (Status Code: {err.response.status_code})")
    print(f"   Server says: {err.response.text}")
except requests.exceptions.RequestException as e:
    print(f"❌ A network error occurred: {e}")
finally:
    # Ensure the file is closed
    if 'files' in locals() and files['image']:
        files['image'][1].close()


# --- Step 2: Call the /prompt endpoint repeatedly ---
if verified_image_id:
    print("\n" + "="*35)
    print("STEP 2: SENDING PROMPTS")
    print("(Type 'exit' or 'quit' to stop)")
    print("="*35)

    while True:
        # Get a prompt from the user
        user_prompt = input("\nEnter a prompt > ")

        if user_prompt.lower() in ['exit', 'quit']:
            print("Exiting client.")
            break
        
        try:
            # Prepare the data for the request
            prompt_data = {
                "image_id": verified_image_id,
                "prompt": user_prompt
            }

            print(f"Sending prompt to {PROMPT_URL}...")
            
            # Make the POST request
            response = requests.post(PROMPT_URL, data=prompt_data)
            response.raise_for_status()

            # Print the result
            print("\n--- ROBOBRAIN RESPONSE ---")
            print(response.json())
            print("--------------------------")

        except requests.exceptions.HTTPError as err:
            print(f"❌ Prompt Failed! (Status Code: {err.response.status_code})")
            print(f"   Server says: {err.response.text}")
        except requests.exceptions.RequestException as e:
            print(f"❌ A network error occurred: {e}")

