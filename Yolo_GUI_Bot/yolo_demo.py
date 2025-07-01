from ultralytics import YOLO

model = YOLO(f"runs/detect/train5/weights/best.pt")
results = model("img.jpg", save=True)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox out, puts
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Class probabilities for classification outputs

    print(boxes, masks, keypoints, probs)
