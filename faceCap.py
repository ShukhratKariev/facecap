import cv2
import mediapipe as mp
import os
import sys

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def executable_path():
    """ Get the directory of the executable or script """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

# Save folder (in the same directory as the exe)
output_folder = os.path.join(executable_path(), "faces")
os.makedirs(output_folder, exist_ok=True)

# Video path (resource bundled inside exe or next to exe)
video_path = resource_path("206779_tiny.mp4")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Failed to open video file: {video_path}")
    sys.exit(1)

frame_count = 0
face_id = 0

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin * width), int(bbox.ymin * height), int(bbox.width * width), int(bbox.height * height)

                # Expand a bit around the face
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)

                face_crop = frame[y1:y2, x1:x2]

                # Save face crop
                face_filename = os.path.join(output_folder, f"frame{frame_count}_face{face_id}.jpg")
                cv2.imwrite(face_filename, face_crop)
                face_id += 1

cap.release()
print(f"✅ Done! Saved {face_id} faces in: {output_folder}")