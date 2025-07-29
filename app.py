
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image

# Set title
st.title("APTOS DR Classifier 🔍")
st.markdown("Upload a retina fundus image to predict the level of Diabetic Retinopathy.")

# Class names
class_names = [
    "0 - No DR",
    "1 - Mild",
    "2 - Moderate",
    "3 - Severe",
    "4 - Proliferative DR"
]

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load("retino_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Upload image
uploaded_file = st.file_uploader("Upload a fundus image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()
        st.subheader(f"Prediction: {class_names[prediction]}")

    # Grad-CAM visualization
    st.markdown("---")
    st.subheader("Grad-CAM Heatmap 🔥")

    # Convert image for GradCAM
    rgb_img = np.array(image.resize((224, 224))) / 255.0
    cam_image_input = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    targets = [ClassifierOutputTarget(prediction)]
    grayscale_cam = cam(input_tensor=cam_image_input, targets=targets)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    st.image(visualization, caption="Grad-CAM", use_column_width=True)
