import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Watch Detection System",
    page_icon="⌚",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("⌚ Watch Detection System")

st.write(
    "Upload an image of a hand or wrist and "
    "the machine learning model will predict "
    "whether a watch is present."
)


# -----------------------------
# Load Trained Model
# -----------------------------

try:
    model = joblib.load("models/watch_model.pkl")
except:
    st.error(
        "Trained model not found. "
        "Please run main.py first."
    )
    st.stop()


# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    st.write("")

    if st.button("🔍 Predict Watch"):

        # Convert image to OpenCV format
        image_array = np.array(image)

        # Convert RGB to BGR
        image_array = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        # Resize image
        image_array = cv2.resize(
            image_array,
            (64, 64)
        )

        # Flatten image
        image_array = image_array.flatten()

        # Prepare for prediction
        image_array = np.array(
            [image_array]
        )

        # Make prediction
        prediction = model.predict(
            image_array
        )

        # Get probability
        probability = model.predict_proba(
            image_array
        )

        confidence = max(
            probability[0]
        ) * 100


        # -----------------------------
        # Display Result
        # -----------------------------

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.success("⌚ WITH WATCH")

        else:

            st.info("❌ WITHOUT WATCH")


        st.write(
            f"**Confidence: {confidence:.2f}%**"
        )


# -----------------------------
# Project Information
# -----------------------------

st.divider()

st.subheader("About the Project")

st.write(
    "This project uses image preprocessing and "
    "a Support Vector Machine (SVM) classifier "
    "to identify whether a watch is present "
    "in an image."
)

st.write(
    "**Algorithm:** SVM  \n"
    "**Image Size:** 64 × 64  \n"
    "**Classes:** With Watch / Without Watch"
)
