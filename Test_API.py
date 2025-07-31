# client_with_drawing.py

import requests
import os
import re
from PIL import Image, ImageDraw
from Resize import process_and_resize_image

# --- Configuration ---
#SERVER_IP = "192.168.133.66" # Replace with your server's IP
SERVER_URL = f"https://4a1b23a24493.ngrok-free.app/inference/"#"http://{SERVER_IP}:8000/inference/"
IMAGE_PATH = process_and_resize_image("./assets/demo/Keyboard.jpeg", 500)
PROMPT = input("User Prompt: ")
DOT_RADIUS = 6 # ADJUSTABLE: Change the size of the dots here

# --- Helper function to draw on the image ---
def draw_points_on_image(image_path, points, radius):
    """
    Draws points on an image and saves it into a specific directory.
    
    Args:
        image_path (str): The path to the original image.
        points (list): A list of tuples, e.g., [(x1, y1), (x2, y2)].
        radius (int): The radius of the circles to draw.
    """
    try:
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            
            # Define the color for the points
            fill_color = "red"
            
            for point in points:
                x, y = point
                # Define the bounding box for the circle using the provided radius
                bbox = [x - radius, y - radius, x + radius, y + radius]
                draw.ellipse(bbox, fill=fill_color, outline="black")

            # --- MODIFIED SECTION ---
            # Define the output directory and create it if it doesn't exist
            output_dir = "inference_result"
            os.makedirs(output_dir, exist_ok=True)

            # Create the new filename
            base_filename, ext = os.path.splitext(os.path.basename(image_path))
            new_filename = f"{base_filename}_annotated{ext}"

            # Create the full output path inside the new directory
            output_path = os.path.join(output_dir, new_filename)
            
            img.save(output_path)
            print(f"\n✅ Annotated image saved to: {output_path}")
            # Optional: open the image after saving
            # img.show()

    except Exception as e:
        print(f"\n❌ Error drawing on image: {e}")

# --- Main execution block ---
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at path: {IMAGE_PATH}")
else:
    files = {'image': (os.path.basename(IMAGE_PATH), open(IMAGE_PATH, 'rb'), 'image/jpeg')}
    payload = {'text': PROMPT}

    print(f"Sending request to server at {SERVER_URL}...")
    try:
        response = requests.post(SERVER_URL, files=files, data=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n--- ✅ Server Response ---")
        print(f"Thinking: {result.get('thinking')}")
        print(f"Answer: {result.get('answer')}")
        print("---------------------------")

        # --- Parse coordinates from the answer ---
        answer_text = result.get('answer', '')
        # This regex finds all tuples of numbers like (123, 456)
        point_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        extracted_points = re.findall(point_pattern, answer_text)
        
        if extracted_points:
            # Convert the extracted string numbers to integer tuples
            points_to_draw = [(int(x), int(y)) for x, y in extracted_points]
            print(f"\nFound {len(points_to_draw)} points to draw: {points_to_draw}")
            # Call the drawing function, passing the adjustable radius
            draw_points_on_image(IMAGE_PATH, points_to_draw, DOT_RADIUS)
        else:
            print("\nNo coordinates found in the answer to draw.")

    except requests.exceptions.RequestException as e:
        print(f"\n--- ❌ Network Error ---")
        print(f"An error occurred: {e}")
