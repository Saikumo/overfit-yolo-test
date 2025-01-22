from ultralytics import YOLO

# Load a model
model = YOLO("/kaggle/working/overfit-yolo-test/yolo11n.yaml")  # build a new model from YAML

# Train the model
results = model.train(epochs=1000, imgsz=640)