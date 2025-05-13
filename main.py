import sys
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import shutil
import uuid
import json
import glob
import time

app = FastAPI()

# Allow CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-video/")
async def process_uploaded_video(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        return JSONResponse(status_code=400, content={"detail": "Please upload a valid video file (.mp4, .avi, .mov)"})

    upload_id = str(uuid.uuid4())
    video_filename = f"uploaded_video_{upload_id}{os.path.splitext(file.filename)[1]}"
    video_path = os.path.join(os.path.dirname(__file__), video_filename)
    output_folder = os.path.join(os.path.dirname(__file__), f"unique_faces_{upload_id}")
    zip_filename = f"face_snapshots_{upload_id}.zip"
    zip_path = os.path.join(os.path.dirname(__file__), zip_filename)
    zip_base = os.path.splitext(zip_path)[0]

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        os.makedirs(output_folder, exist_ok=True) # Ensure output folder exists

        script_path = os.path.join(os.path.dirname(__file__), "faceCap.py")
        python_executable = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe") # Adjust if your venv is elsewhere

        result = subprocess.run(
            [python_executable, script_path, video_path, output_folder],
            capture_output=True,
            text=True,
            check=True
        )

        try:
            output = json.loads(result.stdout)
            print(f"faceCap.py output: {output}")
        except json.JSONDecodeError:
            print(f"Warning: faceCap.py output was not valid JSON:\n{stdout.decode()}")

        print(f"Attempting to create ZIP archive:")
        print(f"  - Base name: {zip_base}")
        print(f"  - Format: zip")
        print(f"  - Root dir: {output_folder}")

        try:
            shutil.make_archive(zip_base, 'zip', output_folder)
            time.sleep(0.5)  # Increased delay for testing
            print(f"ZIP archive creation successful (hopefully): {zip_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Checking if ZIP exists at: {zip_path}")
        except Exception as e:
            print(f"ERROR during ZIP creation: {e}")
            return JSONResponse(status_code=500, content={"detail": f"Failed to create ZIP archive: {e}"})

        if os.path.exists(zip_path):
            return FileResponse(zip_path, media_type="application/zip", filename=zip_filename)
        else:
            print(f"WARNING: ZIP archive not found at: {zip_path}")
            return JSONResponse(status_code=500, content={"detail": "ZIP archive not found after creation."})

    except subprocess.CalledProcessError as e:
        error_message = e.stderr
        print(f"faceCap.py error:\n{error_message}")
        return JSONResponse(status_code=500, content={"detail": f"faceCap.py processing failed: {error_message}"})
    except Exception as e:
        print(f"Error processing video: {e}")
        return JSONResponse(status_code=500, content={"detail": f"An error occurred: {e}"})
    finally:
        # Clean up the temporary video and the snapshot folder
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        # Do NOT delete the zip_path here