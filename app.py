"""
Single-image shark polygon demo
Gradio 4.39.x  (downgrade: pip install --force-reinstall gradio==4.39.2)
"""


from ultralytics import YOLO
import gradio as gr

MODEL_PATH = "E:/SERRAFINS/shrkid/runs/detect/shark_yolo117/weights/best.pt"

# load once at start
model = YOLO(MODEL_PATH)

def shark_id(image):
    """
    image: numpy array RGB
    returns: annotated RGB numpy array
    """
    results = model.predict(image, conf=0.15, imgsz=640)
    return results[0].plot()#[:, :, ::-1]    BGR → RGB for Gradio

# build interface
demo = gr.Interface(
    fn=shark_id,
    inputs=gr.Image(type="numpy", label="Drop shark image"),
    outputs=gr.Image(type="numpy", label=""),
    title="YOLOv11 Shark Identifier",
)

# launch
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True, show_api=False)

