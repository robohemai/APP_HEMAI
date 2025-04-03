import streamlit as st
import PIL.Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"weights/best.pt")  # Replace with your model path

# Streamlit UI Customization using CSS
st.markdown(
    """
    <style>
        /* Background Color */
        .stApp {
            background-color: #7B1113; /* Deep Blood Red */
        }
        
        /* Custom Text Color */
        h1, h2, h3, h4, h5, h6, p, span, div {
            color: #FFD1D1 !important; /* Soft Pinkish White */
            text-align: center;
        }

        /* File uploader styling */
        .stFileUploader {
            background-color: #FFE6E6; /* Light Red */
            padding: 10px;
            border-radius: 10px;
        }

        /* Center the logo */
        .stImage {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        /* Uploaded image styling */
        .uploadedImage {
            display: flex;
            justify-content: center;
        }

        /* Container styling */
        .stMarkdown {
            background-color: #640D0D; /* Dark Red */
            padding: 15px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo (Centered)
st.image(r"assets/LOGO_HEMAI.png", use_column_width=True)

# Title
st.title("🔬 Hematological Evaluation Through Machine Learning Analysis")

# Disclaimer
st.markdown(
    """
    **DISCLAIMER:**  
    HEMAI only classifies 4 classes: **Healthy Blood**, **Hemolytic Anemia**,  
    **Sickle Cell Anemia**, and **Thalassemia Anemia**.
    """,
    unsafe_allow_html=True
)

# Upload image
uploaded_image = st.file_uploader(
    "📷 Please upload an image of a blood smear from a microscope view (insert zoom specifications).",
    type=["jpg", "png", "jpeg"]
)

if uploaded_image is not None:
    # Read image
    image = PIL.Image.open(uploaded_image)

    # Run inference
    results = model(image)

    # Display results
    st.image(results[0].plot(), caption="🩸 Detected Blood Anomalies", use_column_width=True)
