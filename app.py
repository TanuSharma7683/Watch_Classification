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
    page_title="Watch Classification",
    page_icon="⌚",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⌚ Watch Classification")
st.write("Upload a watch image to classify it using the trained ML model.")


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "watch_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found!")
    st.write("Expected model location:")
    st.code(MODEL_PATH)
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("Error loading the trained model.")
    st.write(str(e))
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

    # Open uploaded image
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    # Convert PIL image to OpenCV format
    image_array = np.array(image)

    # Convert RGB to BGR for OpenCV
    if len(image_array.shape) == 3:
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_cv = image_array

    # Resize image
    image_resized = cv2.resize(
        image_cv,
        (128, 128)
    )

    # Convert image to RGB
    image_rgb = cv2.cvtColor(
        image_resized,
        cv2.COLOR_BGR2RGB
    )

    # Normalize pixel values
    image_normalized = image_rgb / 255.0

    # Flatten image
    image_flattened = image_normalized.flatten()

    # Reshape for model
    image_input = image_flattened.reshape(1, -1)

    # Prediction button
    if st.button("🔍 Classify Watch"):

        try:

            prediction = model.predict(image_input)

            result = prediction[0]

            st.success("Classification completed!")

            st.subheader("Prediction")

            st.write(f"**Result:** {result}")

            # Confidence, if model supports probability prediction
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    image_input
                )

                confidence = np.max(probabilities) * 100

                st.write(
                    f"**Confidence:** {confidence:.2f}%"
                )

                st.progress(
                    int(confidence)
                )

        except Exception as e:

            st.error("Prediction error!")
            st.write(str(e))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Watch Classification System | Python + Machine Learning + Streamlit"
)
```

### ⚠️ One important thing

There is one thing we need to verify: **your trained model's expected image size/features**.

The code above assumes your model was trained using:

```text
128 × 128 image
```

and flattened into features.

Since your **`main.py` already works locally**, the safest approach is actually to use the **same image-processing code from your working `main.py`** in `app.py`. Otherwise, the model may load successfully but give a feature-shape error.

### Now do this

1. Replace your entire `app.py` with the code above.
2. Save it.
3. Test locally:

```text
streamlit run app.py
```

4. If it works locally, commit `app.py` to GitHub.
5. Let Streamlit redeploy.

If you get an error such as:

```text
X has 49152 features, but ...
```

**don't worry**—send me that error. It means we need to match the preprocessing used by your trained model.
