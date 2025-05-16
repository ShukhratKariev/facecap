from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import subprocess
import json
import sys

app = FastAPI()

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    video_path = upload_dir / file.filename

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        file.file.close()

    result = subprocess.run(
        [sys.executable, "faceCap.py", str(video_path)],  # 👈 uses same Python as FastAPI
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return JSONResponse(status_code=500, content={"detail": f"faceCap.py error: {result.stderr.strip()}"})

    try:
        return JSONResponse(content=json.loads(result.stdout))
    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"detail": "Invalid JSON output from faceCap.py"})
