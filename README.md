# Hello
My first repository on GitHub.
ROAD PROJECT CODE:
from flask import Flask, Response
from ultralytics import YOLO
import cv2
import time

# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "best.pt"   # your trained pothole model
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# -----------------------------
# Initialize Flask App
# -----------------------------

app = Flask(__name__)

# -----------------------------
# Load YOLO Model
# -----------------------------

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model Loaded Successfully")

# -----------------------------
# Initialize Camera
# -----------------------------

camera = cv2.VideoCapture(CAMERA_INDEX)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not camera.isOpened():
    print("Error: Camera not accessible")
    exit()

# -----------------------------
# Frame Generator Function
# -----------------------------

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run detection
        results = model(frame, conf=CONFIDENCE_THRESHOLD)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                # Class Names (adjust if you trained multiple classes)
                if class_id == 0:
                    label_name = "Pothole"
                elif class_id == 1:
                    label_name = "Speed Breaker"
                else:
                    label_name = "Object"

                label = f"{label_name} {confidence:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2)

        # Encode frame for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -----------------------------
# Web Routes
# -----------------------------

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Smart Road Monitoring System</title>
        </head>
        <body style="text-align:center;">
            <h2>Live Pothole & Speed Breaker Detection</h2>
            <img src="/video_feed" width="800">
        </body>
    </html>
    """
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------
# Run Server
# -----------------------------

if __name__ == "__main__":
    print("Starting Web Server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
    
    
FOR RUN:
    python pothole_web.py
FOR ACTIVATION:
    source ~/pothole_env/bin/activate
https://github.com/user-attachments/assets/16374fbf-77c4-49f5-ac8d-eee5c56a7b6c


    
