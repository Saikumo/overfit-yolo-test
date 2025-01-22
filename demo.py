from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data="/kaggle/working/overfit-yolo-test/data.yaml",epochs=1000, imgsz=640)