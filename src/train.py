from ultralytics import YOLO
import yaml
import cv2

# COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on COCO8 example (should be switched to produce images later)
# coco8 = None
# try:
#     with open("/data/coco8.yaml", "r") as file:
#         coco8 = yaml.safe_load(file)
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")

results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference
img = cv2.imread("/Users/swedenagu/cs131_project/data/freshApple(1).jpg")
cv2.imshow("image", img)

# results = model
