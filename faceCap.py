import cv2
import mediapipe as mp
import os
import sys
import face_recognition
import numpy as np
import json

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def script_directory():
    """ Get the directory of the script """
    return os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Error: Video path and output folder path must be provided as arguments."}))
        sys.exit(1)

    video_path = sys.argv[1]
    output_folder = sys.argv[2]
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(json.dumps({"error": f"Failed to open video file: {video_path}"}))
        sys.exit(1)

    frame_skip = 10  # Adjust as needed
    frame_count = 0
    saved_faces_count = 0

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            height, width, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)

            if results.detections:
                for i, detection in enumerate(results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(width, x + w + padding)
                    y2 = min(height, y + h + padding)

                    face_crop_bgr = frame[y1:y2, x1:x2]
                    filename = f"face_{frame_count}_detection_{i}.jpg"
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, face_crop_bgr)
                    saved_faces_count += 1

    cap.release()

    print(json.dumps({"status": "processing_done", "saved_faces_count": saved_faces_count}))