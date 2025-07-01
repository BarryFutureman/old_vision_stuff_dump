from ultralytics import YOLO

model = YOLO('yolov8n.yaml').load('my_pretrained/yolov8n.pt')

# Train the model
model.train(data='coco128.yaml', task="detect", pretrained=True, epochs=10, lr0=0.9, lrf=1, imgsz=640, workers=0)
# lr0=0.000002, lrf=0.000002,
