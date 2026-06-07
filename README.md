# Smart Crop Recommendation System

A Streamlit web app that recommends a suitable crop from soil nutrient values and weather conditions. The app uses a trained scikit-learn pipeline saved in `crop_pipeline.pkl` and a label encoder saved in `label_encoder.pkl`.

## Live Deployment Option

The easiest way to make this project live for interviewers is Streamlit Community Cloud.

1. Create a GitHub repository and upload these project files:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `runtime.txt`
   - `crop_pipeline.pkl`
   - `label_encoder.pkl`
   - `README.md`
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Sign in with GitHub.
4. Click **Create app**.
5. Select your repository.
6. Set the main file path to `app.py`.
7. Open **Advanced settings** and choose Python 3.12.
8. Click **Deploy**.

After deployment, Streamlit will give you a public URL that anyone can open.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Inputs

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

## Interview Demo Tips

- Keep the live app URL in your resume or project portfolio.
- Show that the app uses a complete ML workflow: data preprocessing, label encoding, model training, saved pipeline, and web deployment.
- Mention that `st.cache_resource` is used so the model loads efficiently in production.
- Mention that deployment uses Python 3.12 and scikit-learn 1.5.1 to match the saved model artifact.
