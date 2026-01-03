import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# Step 1: Configure Page
st.set_page_config(page_title="AI Image Classifier", layout="centered")
st.title("BSD3513: Computer Vision Image Classifier")

# Step 3: Force CPU Settings
device = torch.device("cpu")

# Step 4: Load Pre-trained ResNet18
@st.cache_resource
def load_model():
    # Try to load a pretrained ResNet18 compatible with multiple torchvision versions
    try:
        model = models.resnet18(pretrained=True)
    except TypeError:
        # Newer torchvision uses the `weights` enum
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.to(device)
    model.eval()
    return model

model = load_model()

# Step 5: Preprocessing Transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet labels (Standard for ResNet)
try:
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
except FileNotFoundError:
    st.error("Please ensure 'imagenet_classes.txt' is in the directory.")
    # Fallback labels to avoid crashes if the file is missing
    categories = [f"Class {i}" for i in range(1000)]

# Step 6: User Interface for File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Step 7: Convert to Tensor and Inference
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    # Step 8: Apply Softmax and Get Top-5 Predictions
    probabilities = F.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Step 9: Visualize with Bar Chart
    results = []
    for i in range(top5_prob.size(0)):
        idx = int(top5_catid[i].item())
        label = categories[idx] if idx < len(categories) else f"Class {idx}"
        results.append({"Class": label, "Probability": top5_prob[i].item()})
    
    df = pd.DataFrame(results)
    st.subheader("Top 5 Predictions")
    st.bar_chart(df.set_index("Class"))
    st.table(df)