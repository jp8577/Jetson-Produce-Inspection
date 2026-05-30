from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO("yolo_FAV.pt")  # trained model
camera = cv2.VideoCapture(0)  # /dev/video0
HTML = """
<!DOCTYPE html><html>
<head><title>Produce Detector</title></head>
<body style="background:#111;color:white;text-align:center">
  <h2>Live Produce Detection</h2>
  <img src="/video" width="800"/>
</body></html>
"""

def generate_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()  # draws boxes + labels automatically

        _, buffer = cv2.imencode('.jpg', annotated)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

