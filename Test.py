from inference import SimpleInference
from Resize import process_and_resize_image

model = SimpleInference("BAAI/RoboBrain2.0-3B")

# Example:
Object = input("Object ID: ")

prompt = input("User Prompt: ")

image = process_and_resize_image("./assets/demo/Pemanas2.png", 500)

pred_2 = model.inference(Object, image, task="verify", plot=False, enable_thinking=False, do_sample=True)

print(f"Comparison result:\n{pred_2['answer']}")

if (pred_2['answer'] == "same"):
    pred_3 = model.inference(prompt, image, task="pointing", plot=False, enable_thinking=False, do_sample=True)
    print(f"Prediction:\n{pred_3['answer']}")
else:
    print("There is no object that satisfies the prompt.")
