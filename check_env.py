import torch
import torchvision
import torchaudio
import ultralytics
import cv2
from PIL import Image  # Pillow
import numpy as np
import gradio as gr
import yaml
import tqdm

print(torch.__version__)  # Should print 2.4.1+cu121
print(torch.cuda.is_available())  # Should print True
print(torch.version.cuda)  # Should print 12.1
print(torch.cuda.get_device_name(0))  # Should print NVIDIA GeForce RTX 4050 Laptop GPU
print(torchvision.__version__)  # Should print 0.19.1+cu121
print(torchaudio.__version__)  # Should print 2.4.1+cu121
print(ultralytics.__version__)  # Should print 8.3.0
print(cv2.__version__)  # Should print 4.10.0
print(Image.__version__)  # Should print 10.4.0
print(np.__version__)  # Should print 1.26.4
print(gr.__version__)  # Should print 4.44.0
print(yaml.__version__)  # Should print 6.0.2
print(tqdm.__version__)  # Should print 4.66.5