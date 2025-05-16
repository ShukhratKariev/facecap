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

def iou(boxA, boxB):
    # Intersection over Union — basic box similarity
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

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
face_clusters = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_skip != 0:
        frame_idx += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        sharpness = image_sharpness(face_img)
        score = sharpness * w * h

        matched = False
        for cluster in face_clusters:
            if iou(cluster["bbox"], (x, y, w, h)) > 0.4:
                if score > cluster["score"]:
                    cluster["score"] = score
                    cluster["image_base64"] = encode_image(face_img)
                    cluster["bbox"] = (x, y, w, h)
                matched = True
                break

        if not matched:
            face_clusters.append({
                "bbox": (x, y, w, h),
                "score": score,
                "image_base64": encode_image(face_img)
            })

    frame_idx += 1

cap.release()

# Return best face per cluster
output = [cluster["image_base64"] for cluster in face_clusters]
print(json.dumps({"faces": output}))
