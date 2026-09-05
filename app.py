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
    model = joblib.load("watch_model.pkl")
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
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Display image
    st.subheader("Uploaded Image")
    st.image(
        image,
        caption="Input Image",
        width=300
    )

    # Convert PIL image to NumPy array
    image_array = np.array(image)

    # Convert image to 3-channel BGR
    if len(image_array.shape) == 3:

        # Handle RGBA images
        if image_array.shape[2] == 4:
            image_array = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGBA2BGR
            )

        # Handle RGB images
        else:
            image_array = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR
            )

    # If grayscale image
    elif len(image_array.shape) == 2:
        image_array = cv2.cvtColor(
            image_array,
            cv2.COLOR_GRAY2BGR
        )

    # --------------------------------------------------
    # IMAGE RESIZING
    # --------------------------------------------------

    image_array = cv2.resize(
        image_array,
        (64, 64)
    )

    # --------------------------------------------------
    # CONVERT IMAGE INTO FEATURES
    # --------------------------------------------------

    image_array = image_array.flatten()

    image_array = np.array(
        image_array
    ).reshape(1, -1)


    # --------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------

    if st.button("Predict"):

        try:

            prediction = model.predict(
                image_array
            )[0]

            # --------------------------------------------------
            # CONVERT NUMERICAL LABEL TO CLASS NAME
            # --------------------------------------------------

            if prediction == 0:

                result = "With Watch"

                st.success(
                    "Prediction: With Watch"
                )

            elif prediction == 1:

                result = "Without Watch"

                st.warning(
                    "Prediction: Without Watch"
                )

            else:

                result = str(prediction)

                st.info(
                    f"Prediction: {result}"
                )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.write(e)
