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

    # Face embedding and change detection settings
    FACE_COMPARISON_TOLERANCE = 0.6
    EMBEDDING_CHANGE_THRESHOLD = 0.15
    frame_skip = 5

    # Face tracking data
    known_faces = []
    face_id_counter = 0
    frame_count = 0

    saved_faces = []

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
                for detection in results.detections:
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

                    face_crop_rgb = np.ascontiguousarray(image_rgb[y1:y2, x1:x2])
                    face_crop_bgr = frame[y1:y2, x1:x2]

                    try:
                        top, right, bottom, left = y1, x2, y2, x1
                        face_locations = [(top, right, bottom, left)]
                        face_encodings = face_recognition.face_encodings(face_crop_rgb, face_locations)

                        if face_encodings:
                            current_embedding = face_encodings[0]
                            matched = False

                            for face_data in known_faces:
                                distance = np.linalg.norm(face_data["embedding"] - current_embedding)
                                if distance < FACE_COMPARISON_TOLERANCE:
                                    matched = True
                                    change = np.linalg.norm(face_data["last_saved_embedding"] - current_embedding)
                                    if change > EMBEDDING_CHANGE_THRESHOLD:
                                        face_data["last_saved_embedding"] = current_embedding
                                        face_data["embedding"] = current_embedding
                                        filename = f"face_{face_data['id']}_frame{frame_count}.jpg"
                                        filepath = os.path.join(output_folder, filename)
                                        cv2.imwrite(filepath, face_crop_bgr)
                                        saved_faces.append(filename)
                                    break

                            if not matched:
                                face_id_counter += 1
                                known_faces.append({
                                    "id": face_id_counter,
                                    "embedding": current_embedding,
                                    "last_saved_embedding": current_embedding
                                })
                                filename = f"face_{face_id_counter}_frame{frame_count}.jpg"
                                filepath = os.path.join(output_folder, filename)
                                cv2.imwrite(filepath, face_crop_bgr)
                                saved_faces.append(filename)

                    except Exception as e:
                        continue  # Skip face on error

    cap.release()

    # You can optionally output some summary info as JSON if needed
    print(json.dumps({"status": "processing_done", "saved_faces_count": len(saved_faces)}))