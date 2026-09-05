
import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Watch Detection",
    page_icon="⌚",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⌚ Watch Detection Using Machine Learning")

st.write(
    "Upload an image to check whether it contains a watch."
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

try:

    model = joblib.load("models/watch_model.pkl")

except Exception as e:

    st.error("Unable to load the trained model.")

    st.write(e)

    st.stop()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        width=300
    )


    # --------------------------------------------------
    # CONVERT PIL IMAGE TO OPENCV FORMAT
    # --------------------------------------------------

    image = np.array(image)

    # Convert RGB to BGR
    if len(image.shape) == 3:

        if image.shape[2] == 4:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR
            )

        else:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )


    # --------------------------------------------------
    # RESIZE IMAGE
    # --------------------------------------------------

    image = cv2.resize(
        image,
        (64, 64)
    )


    # --------------------------------------------------
    # FLATTEN IMAGE
    # --------------------------------------------------

    image = image.flatten()


    # --------------------------------------------------
    # CONVERT TO 2D ARRAY
    # --------------------------------------------------

    image = np.array([image])


    # --------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------

    if st.button("Predict"):

        try:

            prediction = model.predict(image)

            predicted_class = prediction[0]


            # --------------------------------------------------
            # LABEL MAPPING
            # --------------------------------------------------

            if predicted_class == 1:

                st.success(
                    "Prediction: WITH WATCH"
                )

            elif predicted_class == 0:

                st.warning(
                    "Prediction: WITHOUT WATCH"
                )

            else:

                st.info(
                    f"Prediction: {predicted_class}"
                )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.write(e)

