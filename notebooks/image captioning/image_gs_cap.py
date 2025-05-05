import requests
import base64
import json
import os
import pandas as pd

# Define Ollama API URL
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Path to the folder containing images
image_folder = "glass shattered1"  # Change this to your actual folder path

# Output CSV file
output_csv = "gs_captions_1.csv"

# Updated Prompt for Structured Captioning
prompt_text = """
Caption the image strictly in this format:
'A {color} car with {severity} shattered glass on the {location(s) of damage}.'

- Identify the **car's color** (e.g., red, blue, black, silver).
- Identify the **severity of the shattered glass** (light, moderate, severe).
- Use **"shattered glass"** as the only damage type.
- Identify the **location of the shattered glass** from the following:
  - **Front:** Front windshield
  - **Rear:** Rear windshield
  - **Driver’s Side:** Driver’s side window, driver’s side rear window
  - **Passenger’s Side:** Passenger’s side window, passenger’s side rear window
- If there are **multiple shattered glass locations**, list all using "and."
- If all shattered areas have the **same severity**, mention severity once; otherwise, specify per location.
- Do **not** add extra details, explanations, or formatting beyond the required structure.
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
