from inference import SimpleInference
from Resize import process_and_resize_image

model = SimpleInference("BAAI/RoboBrain2.0-3B")

# Example:
prompt = input("User Prompt: ")

image = process_and_resize_image("./assets/demo/Keyboard.jpeg", 500)

pred = model.inference(prompt, image, task="pointing", plot=True, enable_thinking=False, do_sample=True)
print(f"Prediction:\n{pred}")
