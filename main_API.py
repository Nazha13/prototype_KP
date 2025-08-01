# main_api_with_pyngrok.py

import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pyngrok import ngrok, conf
from typing import Union

# Import your custom class from the inference.py file
from inference import SimpleInference

# --- FastAPI Application Setup ---
app = FastAPI(
    title="RoboBrain Inference API",
    description="Access the RoboBrain 2.0 model over the internet via ngrok.",
    version="2.0.0"
)

# --- Model Loading ---
print("Initializing server and loading model...")
model = SimpleInference("BAAI/RoboBrain2.0-3B")
print("Model loaded. Server is ready.")

# --- API Endpoints (No changes here) ---
@app.get("/")
def root():
    return {"message": "Welcome to the RoboBrain API. Send a POST request to /inference/ to use the model."}

@app.post("/inference/")
async def run_inference(
    text: str = Form(...),
    image: UploadFile = File(...),
    do_sample: bool = Form(True),
    temperature: float = Form(0.7)
):
    TEMP_DIR = "temp_images"
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_image_path = os.path.join(TEMP_DIR, image.filename)
    
    try:
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        absolute_image_path = os.path.abspath(temp_image_path)
        print(f"Received request. Processing image at: {absolute_image_path}")

        result = model.inference(
            text=text,
            image=absolute_image_path,
            task="pointing",
            plot=True,
            enable_thinking=False,
            do_sample=do_sample,
            temperature=temperature
        )
        
        return result

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if os.path.exists(TEMP_DIR) and not os.listdir(TEMP_DIR):
            os.rmdir(TEMP_DIR)

# --- Main execution block to start the server and ngrok tunnel ---
if __name__ == "__main__":
    # Get your ngrok authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
    # It's recommended to set this as an environment variable for security,
    # but we will ask for it as input for simplicity.
    NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN")
    if NGROK_AUTHTOKEN is None:
        NGROK_AUTHTOKEN = input("Please enter your ngrok authtoken: ")
    
    try:
        # Configure ngrok with your authtoken
        conf.get_default().auth_token = NGROK_AUTHTOKEN

        # Start an ngrok tunnel to the uvicorn server
        # The 'public_url' variable will hold your public URL
        public_url = ngrok.connect(8000, domain="balanced-vaguely-mastodon.ngrok-free.app")
        
        print("====================================================================")
        print(f"✅ Your server is live!")
        print(f"✅ Public URL: {public_url}")
        print("You can now use this URL in your client script from any network.")
        print("====================================================================")

        # Start the Uvicorn server, which runs the FastAPI app
        uvicorn.run(app, host="0.0.0.0", port=8000)

    except Exception as e:
        print(f"❌ An error occurred with ngrok: {e}")
        print("Please ensure your ngrok authtoken is correct and that ngrok is not already running.")

