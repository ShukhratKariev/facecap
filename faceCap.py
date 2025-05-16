import sys
import cv2
import os
import json
import base64
import numpy as np

def image_sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

if len(sys.argv) < 2:
    print("No video file path provided", file=sys.stderr)
    sys.exit(1)

video_path = sys.argv[1]
if not os.path.exists(video_path):
    print(f"File not found: {video_path}", file=sys.stderr)
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Failed to open video file: {video_path}", file=sys.stderr)
    sys.exit(1)

frame_skip = 5
frame_idx = 0
face_snapshots = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_skip != 0:
        frame_idx += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        sharpness = image_sharpness(face_img)
        score = sharpness * w * h  # prioritize sharp and large faces

        face_snapshots.append({
            "score": score,
            "image_base64": encode_image(face_img)
        })

    frame_idx += 1

cap.release()

# Sort by score and pick top N
top_faces = sorted(face_snapshots, key=lambda x: x["score"], reverse=True)[:3]
output = [f["image_base64"] for f in top_faces]

print(json.dumps({"faces": output}))
