import cv2
import json
import time
from pathlib import Path
import threading
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import zmq
from google.cloud import storage

# Google Cloud service account configuration
GCP_KEY_PATH = "configs/key.json" 
BUCKET_NAME = "cs131-detections"

def init_gcs_bucket():
    storage_client = storage.Client.from_service_account_json(GCP_KEY_PATH)
    return storage_client.bucket(BUCKET_NAME)

def upload_to_gcs(bucket, frame, filename):
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return
    blob = bucket.blob(filename)
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
    print(f"Uploaded to GCS: {filename}")

# Initialize bucket connection
bucket = init_gcs_bucket()

# Model and camera setup
model_path = Path(__file__).parents[1] / "runs/detect/train-9/weights/best.pt"
if not Path(model_path).exists():
    raise FileNotFoundError(f"Model not found at: {model_path}")
model = YOLO(str(model_path))

camera = cv2.VideoCapture(0)

# Flask web server configuration
app = Flask(__name__)
latest_frame = None
latest_result = "Waiting for detection..."

HTML = """
<!DOCTYPE html><html>
<head>
  <title>Produce Detector</title>
  <script>
    setInterval(() => {
      fetch('/result')
        .then(r => r.json())
        .then(data => {
          document.getElementById('result').innerText = data.final;
        });
    }, 1000);
  </script>
</head>
<body style="background:#111;color:white;text-align:center;font-family:Arial">
  <h2>Live Produce Detection - Nano 1</h2>
  <img src="/video" width="800"/>
  <h3 id="result" style="color:#00ff99;margin-top:20px;">Waiting for detection...</h3>
</body></html>
"""

def get_best_detection(results):
    best = None
    best_conf = 0.0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best = {
                    "label": model.names[int(box.cls[0])],
                    "confidence": conf
                }
    return best

# ZeroMQ consensus and cloud upload loop
def zmq_loop():
    global latest_frame, latest_result, bucket

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:5555")
    print("Nano 1 ready, waiting for Nano 2 to connect...")

    frame_idx = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        # Inference configuration matched for Jetson performance optimization
        results = model.predict(
            source=0,  # Logitech Brio 101 on /dev/video0
            show=True, # Set to False to prevent X11 display errors on headless Jetsons
            device=0,  # GPU found on device can be enabled with CUDA on Orin Nano
            half=True,  # Model quantized to FP16 precision
            conf=0.5,  # Confidence level
            iou=0.4,  # Intersection on union threshold
            stream=True,
        )
        my_detection = get_best_detection(results)

        annotated = results[0].plot()
        latest_frame = annotated

        socket.send_string(json.dumps(my_detection or {}))
        msg = socket.recv_string()
        other_detection = json.loads(msg)

        my_conf = my_detection["confidence"] if my_detection else 0.0
        other_conf = other_detection.get("confidence", 0.0)

        if my_conf == 0.0 and other_conf == 0.0:
            final = "No produce detected"
        elif my_conf >= other_conf:
            final = f"[Nano 1 wins] {my_detection['label']} ({my_conf:.2f})"
            
            # Save to cloud bucket if Nano 1 wins the consensus
            timestamp = int(time.time())
            filename = f"detections/nano1_frame_{timestamp}_{frame_idx}.jpg"
            upload_to_gcs(bucket, annotated, filename)
        else:
            final = f"[Nano 2 wins] {other_detection['label']} ({other_conf:.2f})"

        latest_result = final
        print(f"Final decision: {final}")
        frame_idx += 1

# Flask routing
def generate_frames():
    global latest_frame
    while True:
        if latest_frame is None:
            continue
        _, buffer = cv2.imencode('.jpg', latest_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result():
    return json.dumps({"final": latest_result})

if __name__ == '__main__':
    t = threading.Thread(target=zmq_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
