from ultralytics import YOLO

model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='coco128.yaml', epochs=100, imgsz=640, workers=0)
