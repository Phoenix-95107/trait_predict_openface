from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os, subprocess, uuid, shutil
import pandas as pd
from section.relationship_empathy import calculate_section1
from section.work_DNA_focus import calculate_section2
from section.creativity_pulse import calculate_section3
from section.stress_resilience import calculate_section4
from typing import List
import uvicorn
from PIL import Image

app = FastAPI()

RESULT_FOLDER = "results"
OPENFACE_BIN = r"openface\FeatureExtraction.exe"

# Ensure result folder exists
os.makedirs(RESULT_FOLDER, exist_ok=True)
def convert_webp_to_jpg(input_path):
    """Convert WEBP image to JPG format, handling both color and grayscale images."""
    # Get the directory and filename without extension
    directory = os.path.dirname(input_path)
    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Create output path with jpg extension
    output_path = os.path.join(directory, f"{filename}.jpg")
    
    # Open and convert the image
    img = Image.open(input_path)
    
    # Check if image is grayscale (mode 'L')
    if img.mode == 'L':
        # For grayscale, directly save as JPG
        img.save(output_path, "JPEG")
    else:
        # For color images, convert to RGB then save as JPG
        img = img.convert('RGB')
        img.save(output_path, "JPEG")
    
    # Remove the original webp file
    os.remove(input_path)
    
    return output_path

@app.post("/analyze")
async def analyze_video(files: List[UploadFile] = File(...)):
    file_id = str(uuid.uuid4())
    upload_folder = os.path.join("uploads", file_id)
    output_path = os.path.join(RESULT_FOLDER, file_id)

    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    try:
        # Save uploaded files
        for file in files:
            file_ext = file.filename.split('.')[-1].lower()
            
            unique_filename = f"{uuid.uuid4()}"
            input_path = os.path.join(upload_folder, f"{unique_filename}.{file_ext}")
            
            with open(input_path, "wb") as f:
                f.write(await file.read())
            
            # Convert webp to jpg if needed
            if file_ext == 'webp':
                input_path = convert_webp_to_jpg(input_path)

        # Run OpenFace
        command = [
            OPENFACE_BIN,
            "-fdir", upload_folder,
            "-out_dir", output_path
        ]
        subprocess.run(command, check=True)

        # Load OpenFace CSVs
        csv_files = f"{output_path}\{file_id}.csv"
        print(pd.read_csv(csv_files))
        section1 = calculate_section1(csv_files)
        section2 = calculate_section2(csv_files)
        section3 = calculate_section3(csv_files)
        section4 = calculate_section4(csv_files)

        return {
            "section1": section1,
            "section2": section2,
            "section3": section3,
            "section4": section4
        }

    finally:
        # Clean up upload folder after processing
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
        if os.path.exists(RESULT_FOLDER):
            shutil.rmtree(RESULT_FOLDER)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
