from ultralytics import YOLO

model = YOLO('my_pretrained/yolov8n.pt')

# Train the model
model.train(data='data.yaml', task="detect", pretrained=True, epochs=25, imgsz=640, workers=0)
# lr0=0.000002, lrf=0.000002,
