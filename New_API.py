# main_api_with_pyngrok.py

import os
import shutil
import uvicorn
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pyngrok import ngrok, conf
from typing import Union

# Import your custom class from the inference.py file
from inference import SimpleInference

# --- Global Settings & Setup ---
VERIFIED_DIR = "verified_images"
os.makedirs(VERIFIED_DIR, exist_ok=True)

app = FastAPI(
    title="RoboBrain Stateful API",
    description="A two-step API: 1. Verify an image and get an ID. 2. Use the ID to send multiple prompts.",
    version="3.0.0"
)

# --- Model Loading ---
print("Initializing server and loading model...")
model = SimpleInference("BAAI/RoboBrain2.0-3B")
print("Model loaded. Server is ready.")

# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Welcome to the Stateful RoboBrain API. Use /verify and /prompt endpoints."}

@app.post("/verify")
async def verify_image_and_get_id(
    object_id: str = Form(..., description="A description of the object to verify in the image."),
    image: UploadFile = File(...)
):
    """
    Verifies if an object is in an image. If successful, saves the image
    and returns a unique image_id for use with the /prompt endpoint.
    """
    # Use a temporary file for initial processing
    temp_upload_path = os.path.join(VERIFIED_DIR, f"temp_{image.filename}")
    
    try:
        with open(temp_upload_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # --- Run Verification ---
        print(f"Verifying object '{object_id}' in image '{image.filename}'...")
        verification_result = model.inference(
            text=object_id,
            image=os.path.abspath(temp_upload_path),
            task="verify",
            plot=False,
            enable_thinking=False,
            do_sample=True
        )

        if verification_result.get("answer") == "same":
            # --- Verification Successful ---
            # Generate a unique ID and a new, permanent path
            image_id = str(uuid.uuid4())
            file_extension = os.path.splitext(image.filename)[1]
            permanent_path = os.path.join(VERIFIED_DIR, f"{image_id}{file_extension}")
            
            # Move the temp file to its permanent location
            os.rename(temp_upload_path, permanent_path)
            
            print(f"Verification successful. Image saved as {image_id}{file_extension}")
            return {"status": "verified", "image_id": image_id}
        else:
            # --- Verification Failed ---
            # Clean up the temporary file and return the error
            os.remove(temp_upload_path)
            print("Verification failed. Sending comedic error.")
            raise HTTPException(
                status_code=404, 
                detail="YOU DARE LIE TO ROBOBRAIN????"
            )
            
    except Exception as e:
        # Clean up temp file on any error
        if os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


@app.post("/prompt")
async def run_prompt_on_verified_image(
    image_id: str = Form(..., description="The unique ID of the previously verified image."),
    prompt: str = Form(..., description="The pointing instruction for the model.")
):
    """
    Runs a pointing task on an image that has already been verified,
    using its unique image_id.
    """
    # Find the image file by its ID, checking common extensions
    image_path = None
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        potential_path = os.path.join(VERIFIED_DIR, f"{image_id}{ext}")
        if os.path.exists(potential_path):
            image_path = potential_path
            break
    
    if not image_path:
        raise HTTPException(status_code=404, detail=f"Image with ID '{image_id}' not found. Please verify the image first.")

    try:
        # --- Run Pointing Task ---
        print(f"Running prompt '{prompt}' on image_id '{image_id}'...")
        pointing_result = model.inference(
            text=prompt,
            image=os.path.abspath(image_path),
            task="pointing",
            plot=True,
            enable_thinking=False,
            do_sample=True
        )
        print("Pointing task complete.")
        return pointing_result

    except Exception as e:
        print(f"An error occurred during the pointing task: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


# --- Main execution block to start the server and ngrok tunnel ---
if __name__ == "__main__":
    NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN")
    if NGROK_AUTHTOKEN is None:
        NGROK_AUTHTOKEN = input("Please enter your ngrok authtoken: ")
    
    try:
        conf.get_default().auth_token = NGROK_AUTHTOKEN
        public_url = ngrok.connect(8000, domain="balanced-vaguely-mastodon.ngrok-free.app")
        
        print("====================================================================")
        print(f"✅ Your server is live!")
        print(f"✅ Public URL: {public_url}")
        print("You can now use this URL in your client script from any network.")
        print("====================================================================")

        uvicorn.run(app, host="0.0.0.0", port=8000)

    except Exception as e:
        print(f"❌ An error occurred with ngrok: {e}")
        print("Please ensure your ngrok authtoken is correct and that ngrok is not already running.")
