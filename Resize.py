from PIL import Image
import os

def process_and_resize_image(input_path, max_size=1024):
    """
    Checks an image's size and resizes it if it's too large,
    saving it with a new name. Returns the path to the processed image.
    """
    try:
        img = Image.open(input_path)
        
        # Check if resizing is needed
        if max(img.size) > max_size:
            print(f"Resizing image: {input_path} (Original size: {img.size})")
            
            # This maintains the aspect ratio
            img.thumbnail((max_size, max_size))
            
            # Create a new filename for the resized image
            directory, filename = os.path.split(input_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_resized{ext}"
            output_path = os.path.join(directory, output_filename)
            
            img.save(output_path)
            print(f"Resized image saved to: {output_path}")
            return output_path
        else:
            # If no resizing is needed, just return the original path
            print(f"Image {input_path} is already within size limits.")
            return input_path
            
    except Exception as e:
        print(f"Error processing image {input_path}: {e}")
        return None