import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tensorflow.keras.models import load_model
import os

# Load models and data
soy_model = load_model('soybean_yield_model.h5')
yield_df = pd.read_csv('yield_df_soybeans.csv')

# App config
st.set_page_config(page_title="AI Yield Estimator", layout="wide")
st.sidebar.title("🌾 Navigation")
page = st.sidebar.radio("Select Mode", ["📊 Soybean (Plant Data)", "🌍 Historical (Climate-based)"])

# Page 1 — Soybean physiological input
if page == "📊 Soybean (Plant Data)":
    st.title("🌱 Soybean Yield Prediction (Physiological Data)")

    col1, col2, col3 = st.columns([5, 1, 1])
    with col3:
        language = st.selectbox("🌐", ["English", "한국어"])

    uploaded_file = st.file_uploader("Upload your soybean data CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("Advanced Soybean Agricultural Dataset.csv")

    with st.sidebar:
        st.header("ℹ️ How to Use")
        st.markdown("""
        - Upload CSV or input data manually.
        - Required: Plant height, leaf width, node count, biomass, etc.
        - Output: Yield prediction per plant (in grams).
        """)

    st.subheader("🌿 Manual Input")
    with st.form("manual_input"):
        p_height = st.number_input("Plant Height (cm)", 0.0, 300.0, 0.0)
        l_width = st.number_input("Leaf Width (cm)", 0.0, 50.0, 0.0)
        n_nodes = st.number_input("Number of Nodes", 0, 50, 0)
        pods = st.number_input("Pod Count", 0, 1000, 0)
        flowers = st.number_input("Flowering Days", 0, 200, 0)
        g_size = st.number_input("Grain Size Index", 0.0, 100.0, 0.0)
        biomass = st.number_input("Biomass (g)", 0.0, 1000.0, 0.0)
        moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 0.0)
        submit = st.form_submit_button("Predict Yield")
        if submit:
            input_data = np.array([[p_height, l_width, n_nodes, pods, flowers, g_size, biomass, moisture]])
            pred = soy_model.predict(input_data)
            st.success(f"🌾 Predicted Yield: {pred[0][0]:.2f} grams")

    if st.button("📊 Predict for Entire Dataset"):
        input_features = df.iloc[:, :-1] if 'yield' in df.columns[-1].lower() else df
        preds = soy_model.predict(input_features)
        df['Predicted_Yield'] = preds
        st.dataframe(df.head())
        st.plotly_chart(px.histogram(df, x="Predicted_Yield", nbins=20))
        st.metric("Average Yield", f"{df['Predicted_Yield'].mean():.2f} g")
        st.metric("Std. Dev.", f"{df['Predicted_Yield'].std():.2f} g")

# Page 2 — Historical data (Climate)
elif page == "🌍 Historical (Climate-based)":
    st.title("🌍 Historical Soybean Yield Explorer (Climate Data)")

    st.markdown("Country-level yield vs rainfall, temperature, and pesticide use.")
    st.dataframe(yield_df.head())

    fig = px.scatter(
        yield_df[yield_df["Item"] == "Soybeans"],
        x="average_rain_fall_mm_per_year",
        y="hg/ha_yield",
        size="pesticides_tonnes",
        color="avg_temp",
        hover_name="Area",
        animation_frame="Year",
        title="Soybean Yield vs Rainfall & Temperature"
    )
    st.plotly_chart(fig)

    st.markdown("⚠️ *Model under development. Climate forecasting and prediction to be added soon.*")
