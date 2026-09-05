import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Watch Classification",
    page_icon="⌚",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⌚ Watch Classification Using Machine Learning")
st.write("Upload an image to classify whether it is With Watch or Without Watch.")


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

try:
    model = joblib.load("models/watch_model.pkl")
except Exception as e:
    st.error("Model file could not be loaded.")
    st.write(e)
    st.stop()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a watch image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# IMAGE PROCESSING AND PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.subheader("Uploaded Image")
    st.image(image, caption="Input Image", width=300)

    # Convert PIL image to OpenCV/Numpy format
    image_array = np.array(image)

    # Convert RGB/RGBA image to BGR/Grayscale-compatible format
    if len(image_array.shape) == 3:

        if image_array.shape[2] == 4:
            image_array = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGBA2BGR
            )
        else:
            image_array = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR
            )

    # Resize image
    image_array = cv2.resize(image_array, (64, 64))

    # Flatten image into feature vector
    image_array = image_array.flatten()

    # Convert to NumPy array and reshape
    image_array = np.array(image_array).reshape(1, -1)

    # Prediction button
    if st.button("Predict"):

        try:
            prediction = model.predict(image_array)[0]

            # --------------------------------------------------
            # CONVERT MODEL LABEL TO ACTUAL CLASS NAME
            # --------------------------------------------------

            if prediction == 0:
                result = "With Watch"
            elif prediction == 1:
                result = "Without Watch"
            else:
                result = str(prediction)

            # Display result
            st.subheader("Classification Result")

            if prediction == 0:
                st.success("Prediction: With Watch")
            elif prediction == 1:
                st.warning("Prediction: Without Watch")
            else:
                st.info(f"Prediction: {result}")

        except Exception as e:
            st.error("Prediction failed.")
            st.write(e)
