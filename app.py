from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import sklearn
import streamlit as st


MODEL_PATH = Path(__file__).with_name("crop_pipeline.pkl")
ENCODER_PATH = Path(__file__).with_name("label_encoder.pkl")

PRESETS = {
    "Rice-ready field": {
        "nitrogen": 90.0,
        "phosphorus": 42.0,
        "potassium": 43.0,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.0,
    },
    "Warm dry soil": {
        "nitrogen": 35.0,
        "phosphorus": 30.0,
        "potassium": 28.0,
        "temperature": 31.5,
        "humidity": 48.0,
        "ph": 6.8,
        "rainfall": 72.0,
    },
    "High nutrient farm": {
        "nitrogen": 115.0,
        "phosphorus": 58.0,
        "potassium": 52.0,
        "temperature": 24.4,
        "humidity": 76.0,
        "ph": 6.2,
        "rainfall": 128.0,
    },
}


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        pipeline = pickle.load(model_file)
    with ENCODER_PATH.open("rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    return pipeline, label_encoder


def build_features(values):
    return np.array(
        [
            [
                values["nitrogen"],
                values["phosphorus"],
                values["potassium"],
                values["temperature"],
                values["humidity"],
                values["ph"],
                values["rainfall"],
            ]
        ]
    )


def predict_crop(values):
    pipeline, label_encoder = load_artifacts()
    features = build_features(values)
    raw_prediction = pipeline.predict(features)[0]
    crop_name = label_encoder.inverse_transform([raw_prediction])[0]

    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        probability_values = pipeline.predict_proba(features)[0]
        crop_labels = label_encoder.inverse_transform(np.arange(len(probability_values)))
        probabilities = pd.DataFrame(
            {"Crop": crop_labels, "Confidence": probability_values}
        ).sort_values("Confidence", ascending=False)

    return crop_name, probabilities


def nutrient_score(values):
    total = values["nitrogen"] + values["phosphorus"] + values["potassium"]
    return min(total / 240, 1.0)


def climate_score(values):
    humidity_score = 1 - abs(values["humidity"] - 65) / 65
    temperature_score = 1 - abs(values["temperature"] - 25) / 25
    rainfall_score = min(values["rainfall"] / 200, 1.0)
    return max((humidity_score + temperature_score + rainfall_score) / 3, 0)


st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .hero {
            padding: 28px 30px;
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(46, 125, 50, 0.92), rgba(3, 105, 161, 0.86)),
                url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1600&q=80");
            background-size: cover;
            background-position: center;
            color: white;
            margin-bottom: 22px;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            letter-spacing: 0;
        }

        .hero p {
            max-width: 680px;
            margin: 10px 0 0;
            font-size: 1.02rem;
        }

        .result-card {
            padding: 24px;
            border-radius: 8px;
            border: 1px solid #d7e7d0;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(31, 41, 51, 0.08);
        }

        .result-label {
            color: #52606d;
            font-size: 0.88rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .result-crop {
            color: #1b5e20;
            font-size: 2.4rem;
            font-weight: 800;
            margin-top: 4px;
        }

        .small-note {
            color: #52606d;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <h1>Smart Crop Recommendation System</h1>
        <p>
            Explore soil nutrients, climate values, and pH levels to generate an
            ML-powered crop recommendation with confidence scores.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model Details")
    st.write("Algorithm: Random Forest Classifier")
    st.write("Pipeline: StandardScaler + trained model")
    st.write("Features: N, P, K, temperature, humidity, pH, rainfall")
    st.write(f"scikit-learn: {sklearn.__version__}")
    st.divider()
    st.caption("Choose a preset, then fine-tune the field values.")

preset_name = st.selectbox("Start with a sample field", list(PRESETS.keys()))
preset = PRESETS[preset_name]

input_tab, insight_tab = st.tabs(["Field Inputs", "Live Field Insights"])

with input_tab:
    soil_col, climate_col, action_col = st.columns([1.1, 1.1, 0.9])

    with soil_col:
        st.subheader("Soil Nutrients")
        nitrogen = st.slider("Nitrogen (N)", 0.0, 150.0, preset["nitrogen"], 1.0)
        phosphorus = st.slider("Phosphorus (P)", 0.0, 150.0, preset["phosphorus"], 1.0)
        potassium = st.slider("Potassium (K)", 0.0, 210.0, preset["potassium"], 1.0)
        ph = st.slider("Soil pH Level", 0.0, 14.0, preset["ph"], 0.1)

    with climate_col:
        st.subheader("Climate Conditions")
        temperature = st.slider(
            "Temperature (deg C)", 0.0, 50.0, preset["temperature"], 0.1
        )
        humidity = st.slider("Humidity (%)", 0.0, 100.0, preset["humidity"], 0.1)
        rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, preset["rainfall"], 0.1)

    values = {
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }

    with action_col:
        st.subheader("Field Snapshot")
        st.metric("NPK Total", f"{nitrogen + phosphorus + potassium:.0f}")
        st.metric("Climate Score", f"{climate_score(values) * 100:.0f}%")
        st.metric("pH Category", "Neutral" if 6 <= ph <= 7.5 else "Needs review")
        predict_clicked = st.button(
            "Recommend Best Crop", type="primary", use_container_width=True
        )

    if predict_clicked:
        try:
            result, probabilities = predict_crop(values)

            result_col, chart_col = st.columns([0.9, 1.1])
            with result_col:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Recommended crop</div>
                        <div class="result-crop">{result.title()}</div>
                        <div class="small-note">
                            This recommendation is based on your selected soil and climate values.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with chart_col:
                if probabilities is not None:
                    top_predictions = probabilities.head(5).copy()
                    top_predictions["Confidence"] = top_predictions["Confidence"] * 100
                    st.subheader("Top Crop Matches")
                    st.bar_chart(
                        top_predictions.set_index("Crop"),
                        y="Confidence",
                        color="#2e7d32",
                    )
                    st.caption(
                        f"Best match confidence: {top_predictions.iloc[0]['Confidence']:.1f}%"
                    )
                else:
                    st.info("This model does not expose probability scores.")

        except FileNotFoundError:
            st.error(
                "Model files are missing. Please include crop_pipeline.pkl and label_encoder.pkl."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with insight_tab:
    values = {
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Nutrient Balance", f"{nutrient_score(values) * 100:.0f}%")
    metric_col2.metric("Climate Suitability", f"{climate_score(values) * 100:.0f}%")
    metric_col3.metric("Rainfall", f"{rainfall:.1f} mm")

    st.subheader("Input Profile")
    profile_df = pd.DataFrame(
        {
            "Feature": [
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Temperature",
                "Humidity",
                "pH",
                "Rainfall",
            ],
            "Value": [
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall,
            ],
        }
    )
    st.bar_chart(profile_df.set_index("Feature"), y="Value", color="#0369a1")

    st.info(
        "Use the preset selector to compare different field conditions, then adjust sliders to see how the profile changes."
    )
