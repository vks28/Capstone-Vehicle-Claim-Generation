import requests
import base64
import json
import os
import pandas as pd

# Define Ollama API URL
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Path to the folder containing images
image_folder = "scratch1"  # Change this to your actual folder path

# Output CSV file
output_csv = "scratch_captions_1.csv"

# Updated Prompt for Structured Captioning
prompt_text = """
Caption the image strictly in this format:
'A {color} car with {severity} scratch(es) on the {location(s) of damage}.'

- Identify the car's color (e.g., red, blue, black, silver).
- Identify the severity of the scratches (light, moderate, severe).
- Use "scratch" as the only damage type.
- If there are multiple scratches, list all affected locations using insurance-standard terminology:
  - **Front:** Front bumper, hood, front left fender, front right fender.
  - **Side:** Driver’s side door, passenger’s side door, driver’s side rear door, passenger’s side rear door.
  - **Rear:** Rear bumper, trunk, rear left fender, rear right fender.
- If the scratches are in multiple areas, separate locations with "and."
- If all scratches have the same severity, mention severity once; otherwise, specify per location.
- Do not add extra details, explanations, or formatting beyond the required structure.
"""

# Get a list of image files (limit to 220)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:220]

# Store captions in a list
captions_data = []

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # Convert image to base64
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

    # API Request Data
    data = {
        "model": "llama3.2-vision",
        "prompt": prompt_text,
        "images": [encoded_image]
    }

    # Send Request (Streaming Mode)
    response = requests.post(OLLAMA_URL, json=data, stream=True)

    # Process Streaming Response (Merge into One Line)
    caption_parts = []  # Store all response parts

    for line in response.iter_lines():
        if line:
            try:
                json_obj = json.loads(line.decode("utf-8"))
                caption_parts.append(json_obj.get("response", "").strip())  # Append each response part
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")

    # Merge all caption parts into a single line
    caption = " ".join(caption_parts).strip()

    # Store the result
    captions_data.append({"Image": image_path, "Caption": caption})

    print(f"✅ Processed: {image_path} -> {caption}")  # Print for debugging

# Convert to DataFrame and Save to CSV
df = pd.DataFrame(captions_data)
df.to_csv(output_csv, index=False)

print(f"\n✅ Captions saved to: {output_csv}")
