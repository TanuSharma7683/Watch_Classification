import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "watch_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found!")
    st.write("Looking for:")
    st.code(MODEL_PATH)
    st.stop()

model = joblib.load(MODEL_PATH)
