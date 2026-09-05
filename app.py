```python
import streamlit as st
import cv2
import numpy as np
import joblib
import os
from PIL import Image


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Watch Detection System",
    page_icon="⌚",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⌚ Watch Detection System")
st.write("Upload a watch image to classify it.")


# --------------------------------------------------
# FIND TRAINED MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "watch_model.pkl"
)


# --------------------------------------------------
# CHECK MODEL FILE
# --------------------------------------------------

if not os.path.isfile(MODEL_PATH):

    st.error("❌ Trained model not found!")

    st.write("The application is looking for:")

    st.code(MODEL_PATH)

    st.write("Please make sure the file is located at:")

    st.code("models/watch_model.pkl")

    st.stop()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error("❌ Error while loading the trained model.")

    st.code(str(e))

    st.stop()


st.success("✅ Trained model loaded successfully!")


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📁 Upload Watch Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )


    # Convert image to NumPy array

    image_array = np.array(image)


    # Handle RGBA images

    if len(image_array.shape) == 3:

        if image_array.shape[2] == 4:

            image_array = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGBA2RGB
            )


    # Resize image

    image_resized = cv2.resize(
        image_array,
        (128, 128)
    )


    # Normalize pixel values

    image_normalized = (
        image_resized.astype("float32") / 255.0
    )


    # Flatten image

    image_flattened = (
        image_normalized.flatten()
    )


    # Create model input

    image_input = image_flattened.reshape(
        1,
        -1
    )


    # --------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------

    if st.button("🔍 Classify Watch"):

        try:

            prediction = model.predict(
                image_input
            )

            result = prediction[0]


            st.subheader("Classification Result")

            st.success(
                f"⌚ Predicted Class: {result}"
            )


            # --------------------------------------------------
            # CONFIDENCE
            # --------------------------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    image_input
                )

                confidence = (
                    np.max(probabilities) * 100
                )

                st.write(
                    f"Confidence: {confidence:.2f}%"
                )

                st.progress(
                    min(int(confidence), 100)
                )


        except Exception as e:

            st.error("❌ Prediction failed.")

            st.code(str(e))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Watch Detection System | "
    "Python + Machine Learning + Streamlit"
)
```
