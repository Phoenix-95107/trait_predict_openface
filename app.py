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

app = FastAPI()

RESULT_FOLDER = "results"
OPENFACE_BIN = r"openface\FeatureExtraction.exe"

# Ensure result folder exists
os.makedirs(RESULT_FOLDER, exist_ok=True)

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
            file_ext = file.filename.split('.')[-1]
            input_path = os.path.join(upload_folder, f"{uuid.uuid4()}.{file_ext}")
            with open(input_path, "wb") as f:
                f.write(await file.read())

        # Run OpenFace
        command = [
            OPENFACE_BIN,
            "-fdir", upload_folder,
            "-out_dir", output_path
        ]
        subprocess.run(command, check=True)

        # Load OpenFace CSVs
        csv_files = f"{output_path}\{file_id}.csv"
        print(csv_files)
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
