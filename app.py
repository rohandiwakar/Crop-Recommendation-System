from pathlib import Path
import pickle

import numpy as np
import streamlit as st


MODEL_PATH = Path(__file__).with_name("crop_pipeline.pkl")
ENCODER_PATH = Path(__file__).with_name("label_encoder.pkl")


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        pipeline = pickle.load(model_file)
    with ENCODER_PATH.open("rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    return pipeline, label_encoder


def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    pipeline, label_encoder = load_artifacts()
    features = np.array(
        [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
    )
    raw_prediction = pipeline.predict(features)[0]
    return label_encoder.inverse_transform([raw_prediction])[0]


st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
)

st.title("Smart Crop Recommendation System")
st.caption(
    "Enter soil nutrients and weather conditions to get an ML-powered crop suggestion."
)

with st.sidebar:
    st.header("Project Highlights")
    st.write("Model: Random Forest Classifier")
    st.write("Pipeline: StandardScaler + trained classifier")
    st.write("Inputs: N, P, K, temperature, humidity, pH, rainfall")
    st.write("Output: Recommended crop")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Soil Nutrients")
    nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0, step=1.0)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0, step=1.0)
    potassium = st.number_input("Potassium (K)", min_value=0.0, value=43.0, step=1.0)

with col2:
    st.subheader("Climate")
    temperature = st.number_input(
        "Temperature (deg C)", min_value=0.0, value=20.8, step=0.1
    )
    humidity = st.number_input(
        "Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1
    )
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=202.0, step=0.1)

with col3:
    st.subheader("Soil Acidity")
    ph = st.number_input(
        "Soil pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1
    )
    st.info(
        "Tip: pH usually ranges from 0 to 14. Most crops prefer slightly acidic to neutral soil."
    )

if st.button("Recommend Crop", type="primary", use_container_width=True):
    try:
        result = predict_crop(
            nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall
        )
        st.success(f"Recommended Crop: {result.title()}")
    except FileNotFoundError:
        st.error(
            "Model files are missing. Please include crop_pipeline.pkl and label_encoder.pkl."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
