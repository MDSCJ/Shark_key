from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Ensure working directory
    os.chdir("E:/SERRAFINS/shrkid")

    # Load pretrained YOLOv11 medium
    model = YOLO("weights/yolo11m.pt")

    # Fine-tune on your Roboflow dataset
    results = model.train(
        data="E:/SERRAFINS/shrkid/sharks/data.yaml",  # Path to data.yaml
        epochs=100,
        imgsz=640,
        batch=16,  # Reduced for RTX 4060 (8GB VRAM) from 16
        device=0,
        workers=0,
        name="shark_yolo11",
        patience=100,
        optimizer="AdamW",
        augment=True  # Enable augmentations
    )

    # Validate
    model.val()

    # Export best model
    model.export(format="onnx")
    print("Best model saved at: runs/detect/shark_yolo11/weights/best.onnx")