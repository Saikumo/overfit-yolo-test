from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Define custom hyperparameters to disable data augmentation
hyperparameters = {
    "hsv_h": 0.0,        # Disable hue augmentation
    "hsv_s": 0.0,        # Disable saturation augmentation
    "hsv_v": 0.0,        # Disable brightness augmentation
    "degrees": 0.0,      # Disable rotation augmentation
    "translate": 0.0,    # Disable translation augmentation
    "scale": 0.0,        # Disable scale augmentation
    "shear": 0.0,        # Disable shear augmentation
    "flipud": 0.0,       # Disable vertical flip
    "fliplr": 0.0        # Disable horizontal flip
}

# Train the model with custom hyperparameters
results = model.train(data="/kaggle/working/overfit-yolo-test/data.yaml",
                      epochs=1000,
                      imgsz=640,
                      hyp=hyperparameters)
