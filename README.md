# Shark_key
Unlock the mysteries of the ocean with this AI-powered model designed to identify shark species. Built on custom datasets, this project is still evolving, but it’s already making waves in shark conservation and research. Dive in and help shape the future of marine science!

Prerequisites

Before you start, make sure you have Python 3.8+ installed. You’ll also need Git for version control.

Steps to Build the AI Model

Prepare the Data

Visit Roboflow
to annotate your shark images. Roboflow makes it easy to label the shark species in your dataset and export it in a YOLO-compatible format.

Download YOLO

Download the relevant version of YOLO. You can find the official YOLO releases on the Ultralytics GitHub
. Ensure you pick the version compatible with your setup.

Set Up a Virtual Environment

Create a virtual environment to isolate dependencies. Open a terminal and run:

python -m venv E:\SERRAFINS\sharkenv


To activate the virtual environment, navigate to the folder E:\SERRAFINS and run:

E:\SERRAFINS\sharkenv\Scripts\activate.bat


Install Dependencies

Once the virtual environment is activated, install the necessary libraries by running:

pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.3.0 opencv-python==4.10.0.84 Pillow==10.4.0 numpy==1.26.4 gradio==4.44.0 PyYAML==6.0.2 tqdm==4.66.5


Create the Python Files

You will need three Python files to run, train, and check your environment.

app.py – This file runs the AI model on new images and provides results for shark species identification.

tune.py – This file is used to train and fine-tune the model with your annotated data.

check_env.py – This file checks if all the necessary libraries are installed correctly.
