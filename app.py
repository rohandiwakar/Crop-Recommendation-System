from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import sklearn
import streamlit as st


MODEL_PATH = Path(__file__).with_name("crop_pipeline.pkl")
ENCODER_PATH = Path(__file__).with_name("label_encoder.pkl")
VIDEO_URL = "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260602_150901_c45b90ec-18d7-42ff-90e2-b95d7109e330.mp4"

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


def condition_label(score):
    if score >= 0.78:
        return "Excellent"
    if score >= 0.55:
        return "Good"
    return "Needs tuning"


def ph_category(ph):
    if 6 <= ph <= 7.5:
        return "Neutral"
    if ph < 6:
        return "Acidic"
    return "Alkaline"


def rainfall_category(rainfall):
    if rainfall >= 180:
        return "Heavy"
    if rainfall >= 80:
        return "Moderate"
    return "Low"


def show_ranked_matches(probabilities):
    top_predictions = probabilities.head(5).copy()
    top_predictions["Confidence"] = top_predictions["Confidence"] * 100

    for _, row in top_predictions.iterrows():
        confidence = float(row["Confidence"])
        label_col, value_col = st.columns([0.7, 0.3])
        label_col.write(row["Crop"].title())
        value_col.write(f"{confidence:.1f}%")
        st.progress(min(confidence / 100, 1.0))

    return float(top_predictions.iloc[0]["Confidence"])


def show_input_profile(profile_df):
    max_value = max(float(profile_df["Value"].max()), 1.0)

    for _, row in profile_df.iterrows():
        value = float(row["Value"])
        label_col, value_col = st.columns([0.7, 0.3])
        label_col.write(row["Feature"])
        value_col.write(f"{value:.1f}")
        st.progress(min(value / max_value, 1.0))


st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600;700&display=swap');

        * {
            font-family: 'Inter', sans-serif;
        }

        .main .block-container {
            min-height: calc(100vh - 48px);
            margin: 1.5rem auto;
            padding: 1.5rem;
            max-width: 1180px;
            border-radius: 28px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.26);
            backdrop-filter: blur(16px);
        }

        .stApp {
            background: #ffffff;
        }

        .video-backdrop {
            position: fixed;
            inset: 0;
            z-index: 0;
            overflow: hidden;
            pointer-events: none;
        }

        .video-backdrop video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: saturate(1.08) contrast(1.04);
        }

        .video-backdrop::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(180deg, rgba(7, 19, 12, 0.18), rgba(247, 251, 246, 0.68)),
                radial-gradient(circle at 18% 20%, rgba(255, 255, 255, 0.22), transparent 28%),
                radial-gradient(circle at 80% 12%, rgba(3, 105, 161, 0.18), transparent 30%);
        }

        .main,
        header,
        section[data-testid="stSidebar"] {
            position: relative;
            z-index: 1;
        }

        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(16px);
        }

        .hero {
            padding: 28px 30px;
            border-radius: 22px;
            background:
                linear-gradient(135deg, rgba(17, 24, 39, 0.82), rgba(27, 94, 32, 0.62)),
                rgba(255, 255, 255, 0.08);
            color: white;
            margin-bottom: 22px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 14px 38px rgba(15, 23, 42, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.22);
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(110deg, transparent 0%, rgba(255, 255, 255, 0.18) 48%, transparent 70%);
            transform: translateX(-80%);
            animation: hero-sweep 8s ease-in-out infinite;
        }

        @keyframes hero-sweep {
            0%, 55% { transform: translateX(-85%); }
            100% { transform: translateX(85%); }
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            letter-spacing: 0;
            position: relative;
            z-index: 1;
            max-width: 760px;
        }

        .hero p {
            max-width: 680px;
            margin: 10px 0 0;
            font-size: 1.02rem;
            position: relative;
            z-index: 1;
        }

        .serif-accent {
            font-family: 'Instrument Serif', serif;
            font-style: italic;
            font-weight: 400;
        }

        .about-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 0 0 22px;
        }

        .about-item {
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(255, 255, 255, 0.55);
            box-shadow: 0 8px 24px rgba(31, 41, 51, 0.08);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .about-item:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(31, 41, 51, 0.13);
        }

        .about-kicker {
            color: #52606d;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .about-text {
            color: #1f2933;
            margin-top: 4px;
            font-weight: 650;
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

        .pulse-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 14px 0 20px;
        }

        .pulse-card {
            padding: 16px;
            border-radius: 8px;
            border: 1px solid rgba(46, 125, 50, 0.18);
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 8px 24px rgba(31, 41, 51, 0.08);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .pulse-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(31, 41, 51, 0.13);
        }

        .pulse-label {
            color: #52606d;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .pulse-value {
            color: #1f2933;
            font-size: 1.35rem;
            font-weight: 750;
            margin-top: 4px;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(3, 105, 161, 0.14);
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 6px 20px rgba(31, 41, 51, 0.06);
        }

        div.stButton > button {
            transition: transform 140ms ease, box-shadow 140ms ease;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(46, 125, 50, 0.2);
        }

        @media (max-width: 800px) {
            .main .block-container {
                margin: 0.75rem;
                padding: 1rem;
                min-height: calc(100vh - 24px);
                border-radius: 22px;
            }

            .about-strip {
                grid-template-columns: 1fr;
            }

            .pulse-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="video-backdrop">
        <video autoplay muted loop playsinline>
            <source src="{VIDEO_URL}" type="video/mp4">
        </video>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <h1>Smart Crop Recommendation System for <span class="serif-accent">smarter farming</span></h1>
        <p>
            A clean machine learning demo that turns soil nutrients, climate values,
            and pH levels into a crop recommendation with confidence scores.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="about-strip">
        <div class="about-item">
            <div class="about-kicker">Project</div>
            <div class="about-text">End-to-end ML model deployed as a live web app</div>
        </div>
        <div class="about-item">
            <div class="about-kicker">Built With</div>
            <div class="about-text">Streamlit, scikit-learn, NumPy, pandas</div>
        </div>
        <div class="about-item">
            <div class="about-kicker">Demo Value</div>
            <div class="about-text">Interactive inputs, live insights, and crop confidence</div>
        </div>
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
        st.metric("pH Category", ph_category(ph))
        auto_predict = st.toggle("Auto recommend while sliding", value=False)
        predict_clicked = st.button(
            "Recommend Best Crop", type="primary", use_container_width=True
        )

    st.markdown(
        f"""
        <div class="pulse-grid">
            <div class="pulse-card">
                <div class="pulse-label">Nutrient readiness</div>
                <div class="pulse-value">{condition_label(nutrient_score(values))}</div>
            </div>
            <div class="pulse-card">
                <div class="pulse-label">Climate condition</div>
                <div class="pulse-value">{condition_label(climate_score(values))}</div>
            </div>
            <div class="pulse-card">
                <div class="pulse-label">Rainfall pattern</div>
                <div class="pulse-value">{rainfall_category(rainfall)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if predict_clicked or auto_predict:
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
                    st.subheader("Top Crop Matches")
                    best_confidence = show_ranked_matches(probabilities)
                    st.caption(f"Best match confidence: {best_confidence:.1f}%")
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
    show_input_profile(profile_df)

    st.info(
        "Use the preset selector to compare different field conditions, then adjust sliders to see how the profile changes."
    )
