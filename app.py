import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io

translations = {
    "English": {
        "title": "🌱 Soybean Yield Prediction Web App",
        "instruction": "Enter plant data manually or upload CSV to predict seed yield (g/m²).",
        "dataset_info": "**About Dataset**  \nThe training data was collected from field experiments and agricultural trials. Parameters such as plant height, pods, biomass, sugars, chlorophyll and protein content were measured using standard sensors and lab analysis. Data is from the Kaggle dataset \"Advanced Soybean Agricultural Dataset\" (2025)."
    },
    "한국어": {
        "title": "🌱 대두 수확량 예측 웹 애플리케이션",
        "instruction": "수동으로 입력하거나 CSV 업로드를 통해 수확량(g/m²)을 예측합니다.",
        "dataset_info": "**데이터셋 정보**  \n학습 데이터는 필드 실험 및 농업 시험에서 수집되었습니다. 식물 높이, 꼬투리 수, 생물량, 당도, 엽록소 및 단백질 함량 등은 표준 센서와 실험실 분석을 통해 측정되었습니다. 데이터는 Kaggle의 \"Advanced Soybean Agricultural Dataset\" (2025)에서 가져왔습니다."
    }
}

# Sidebar README/Help Section
with st.sidebar:
    st.markdown("### ℹ️ How to Use This App")
    st.write("**Purpose**: Predict soybean yield based on physiological plant data using AI.")
    st.markdown("**Usage Steps:**")
    st.markdown("1. **Upload** a `.csv` file with correct plant data, or")
    st.markdown("2. **Enter manually** the numeric values of one plant sample.")
    st.markdown("3. Click **Predict** to view results.")
    st.markdown("**Important:**")
    st.markdown("- All inputs must be numeric.")
    st.markdown("- No empty fields allowed.")
    st.markdown("- Keep inputs within realistic value ranges.")
    st.markdown("**Output Info:**")
    st.markdown("- The prediction shows expected yield in **kg/ha**.")
    st.markdown("- You will see warnings if predicted yield is below average.")
    st.markdown("**Coming Soon:** Weather data integration and image input.")

# Mean and std from training set (Kaggle dataset)
mean_vector = np.array([49.96, 144.94, 114.53, 0.4775, 0.6736, 3.9377, 2.3991,
                        36.0421, 35.3491, 0.0721, 2.0201, 0.5076])
std_vector = np.array([3.0523, 20.016, 48.9255, 0.2363, 0.0845, 2.8463, 1.1873,
                       2.4172, 4.9992, 0.0223, 0.2459, 0.2875])

# Top-right language selector using columns
col1, col2, col3 = st.columns([5, 1, 1])
with col3:
    language = st.selectbox("🌐", ["English", "한국어"])

# Styling
st.markdown("""
<style>
.main {
    background-color: #f0f9ff;
    padding: 20px;
    border-radius: 10px;
}
h1, h2, h3, h4, h5, h6 {
    color: #2E8B57;
}
</style>
""", unsafe_allow_html=True)

# Feature order for the model
feature_names = [
    "Plant Height (PH)", "Number of Pods (NP)", "Biological Weight (BW)", "Sugars (Su)",
    "Relative Water Content in Leaves (RWCL)", "ChlorophyllA663", "Chlorophyllb649",
    "Protein Percentage (PPE)", "Weight of 300 Seeds (W3S)", "Leaf Area Index (LAI)",
    "Number of Seeds per Pod (NSP)", "Protein Content (PCO)"
]

# Load the model
model = tf.keras.models.load_model("soybean_yield_model.h5", compile=False)

# Interface texts
T = {
    "title": {
        "English": "🌱 Soybean Seed Yield Prediction",
        "Korean": "🌱 대두 종자 수확량 예측"
    },
    "instruction": {
        "English": "Enter plant data manually or upload CSV to predict seed yield (g/m²).",
        "Korean": "수동으로 입력하거나 CSV 업로드를 통해 수확량(g/m²)을 예측합니다."
    },
    "dataset_info": {
        "English": "**About Dataset**  \nThe training data was collected from field experiments and agricultural trials. Parameters such as plant height, pods, biomass, sugars, chlorophyll and protein content were measured using standard sensors and lab analysis. Data is from the Kaggle dataset \"Advanced Soybean Agricultural Dataset\" (2025).",
        "Korean": "**데이터셋 정보**  \n학습 데이터는 필드 실험 및 농업 시험에서 수집되었습니다. 식물 높이, 꼬투리 수, 생물량, 당도, 엽록소 및 단백질 함량 등은 표준 센서와 실험실 분석을 통해 측정되었습니다. 데이터는 Kaggle의 \"Advanced Soybean Agricultural Dataset\" (2025)에서 가져왔습니다."
    },
    "method": {"English": "Input Method:", "Korean": "입력 방식:"},
    "manual": {"English": "Manual Input", "Korean": "수동 입력"},
    "upload": {"English": "Upload CSV", "Korean": "CSV 업로드"},
    "predict": {"English": "📈 Predict Yield", "Korean": "📈 수확량 예측"},
    "result": {"English": "🌾 Predicted Yield:", "Korean": "🌾 예측된 수확량:"},
    "recommend_low": {"English": "⚠️ Low yield. Improve irrigation, nutrients or leaf area.", "Korean": "⚠️ 낮은 수확량입니다. 관개, 영양 또는 잎 면적을 개선하세요."},
    "recommend_high": {"English": "🌟 Excellent yield. Keep current conditions!", "Korean": "🌟 우수한 수확량입니다. 현재 조건을 유지하세요!"},
    "recommend_medium": {"English": "🔎 Moderate yield. Try optimizing pods and protein.", "Korean": "🔎 평균 수확량입니다. 꼬투리 수와 단백질을 조정해보세요."},
    "warn_very_low": {"English": "❌ Very low yield. Severe stress likely.", "Korean": "❌ 매우 낮은 수확량. 식물 스트레스가 심각할 수 있습니다."},
    "warn_very_high": {"English": "⚠️ Unusually high yield. Check for data entry errors.", "Korean": "⚠️ 비정상적으로 높은 수확량. 데이터 입력 오류를 확인하세요."},
    "average_yield": {"English": "📊 Average Predicted Yield", "Korean": "📊 평균 예측 수확량"},
    "completed": {"English": "✅ Predictions completed.", "Korean": "✅ 예측이 완료되었습니다."},
    "hist_title": {"English": "📊 Yield Distribution", "Korean": "📊 수확량 분포"},
    "hist_desc": {"English": "Distribution of predicted yields across all samples.", "Korean": "모든 샘플의 예측 수확량 분포입니다."},
    "heatmap_title": {"English": "🔥 Yield Heatmap (PH vs NP)", "Korean": "🔥 수확량 히트맵 (식물 높이 vs 꼬투리 수)"},
    "heatmap_desc": {"English": "Higher yields appear in darker cells. Shows relation between plant height and pod count.", "Korean": "어두운 셀일수록 높은 수확량을 나타냅니다. 식물 높이와 꼬투리 수의 관계를 보여줍니다."},
    "interactive_title": {"English": "🧠 Interactive Yield Visualization", "Korean": "🧠 수확량 상호작용 시각화"},
    "interactive_desc": {"English": "Scatter plot of yield vs. height, colored by pods and sized by LAI.", "Korean": "수확량과 식물 높이의 관계를 보여주는 산점도입니다. 꼬투리 수에 따라 색상이 다르고 LAI에 따라 크기가 조정됩니다."},
    "history": {"English": "🗂️ Prediction History", "Korean": "🗂️ 예측 기록"}
}

# Feature descriptions
feature_info = {
    "Plant Height (PH)": {"en": "Total height of the plant (cm)", "ko": "식물의 전체 높이 (cm)"},
    "Number of Pods (NP)": {"en": "Number of pods per plant", "ko": "식물당 꼬투리 수"},
    "Biological Weight (BW)": {"en": "Total biomass weight (g)", "ko": "생물량 무게 (g)"},
    "Sugars (Su)": {"en": "Sugar content in leaves", "ko": "잎의 당도"},
    "Relative Water Content in Leaves (RWCL)": {"en": "Water ratio in leaves", "ko": "잎의 수분 비율"},
    "ChlorophyllA663": {"en": "Chlorophyll A at 663nm", "ko": "엽록소 A (663nm)"},
    "Chlorophyllb649": {"en": "Chlorophyll B at 649nm", "ko": "엽록소 B (649nm)"},
    "Protein Percentage (PPE)": {"en": "Protein percentage in seeds", "ko": "종자 단백질 비율 (%)"},
    "Weight of 300 Seeds (W3S)": {"en": "Weight of 300 seeds (g)", "ko": "300개 종자의 무게 (g)"},
    "Leaf Area Index (LAI)": {"en": "Leaf area index", "ko": "잎 면적 지수"},
    "Number of Seeds per Pod (NSP)": {"en": "Seeds per pod", "ko": "꼬투리당 종자 수"},
    "Protein Content (PCO)": {"en": "Normalized protein content", "ko": "정규화된 단백질 함량"}
}

# Remaining code will follow the updated theme and corrections (not included here for brevity)
# Ensure you continue with the same format in your app.py


feature_names = list(feature_info.keys())

if "history" not in st.session_state:
    st.session_state.history = []

# Main
st.markdown(f"<div class='main'>", unsafe_allow_html=True)
st.title(translations[language]["title"])
st.write(translations[language]["instruction"])
st.markdown(translations[language]["dataset_info"])

input_method = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    input_data = []
    st.markdown("### ✍️ Enter Parameters:")
    for i, feature in enumerate(feature_names):
        key = f"feature_{i}"
        user_input = st.text_input(feature, value="", key=key, placeholder="Enter value")
        try:
            val = float(user_input)
        except:
            val = 0.0
        input_data.append(val)

    if st.button("📈 Predict Yield"):
        X = np.array(input_data).reshape(1, -1)
        X_scaled = (X - mean_vector) / std_vector
        prediction = model.predict(X_scaled)[0][0]

        st.success(f"🌾 Predicted Yield: **{prediction:.2f} g/m²**")

        if prediction < 2500:
            st.error("❌ Very low yield. Severe stress likely.")
        elif prediction > 8000:
            st.error("⚠️ Unusually high yield. Check for data entry errors.")
        elif prediction < 3000:
            st.warning("⚠️ Low yield. Consider improving irrigation or nutrients.")
        elif prediction > 7000:
            st.success("🌟 Excellent yield. Keep up the good practice!")
        else:
            st.info("🔎 Moderate yield. Try optimizing pods and protein.")

else:
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if all(f in df.columns for f in feature_names):
            X = df[feature_names].values
            X_scaled = (X - mean_vector) / std_vector
            predictions = model.predict(X_scaled).flatten()
            df["Predicted Yield (g/m²)"] = predictions

            st.success("✅ Predictions completed.")
            st.dataframe(df.head())

            st.subheader("📊 Yield Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(predictions, bins=30, kde=True, color="green", ax=ax)
            st.pyplot(fig)

            df["PH_bin"] = pd.cut(df["Plant Height (PH)"], bins=10)
            df["NP_bin"] = pd.cut(df["Number of Pods (NP)"], bins=10)
            pivot = df.pivot_table(index="PH_bin", columns="NP_bin", values="Predicted Yield (g/m²)", aggfunc="mean")

            st.subheader("🔥 Yield Heatmap (PH vs NP)")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot, cmap="Greens", annot=True, fmt=".0f", linewidths=0.5, ax=ax2)
            st.pyplot(fig2)

            st.subheader("🧠 Interactive Yield Visualization")
            fig3 = px.scatter(
                df,
                x="Plant Height (PH)",
                y="Predicted Yield (g/m²)",
                color="Number of Pods (NP)",
                size="Leaf Area Index (LAI)",
                hover_data=["Biological Weight (BW)", "Sugars (Su)"]
            )
            fig3.update_layout(height=500)
            st.plotly_chart(fig3)

        else:
            st.error("The uploaded CSV file does not contain the required features.")

# Show prediction history
if st.session_state.history:
    st.subheader(T["history"][lang])
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df.style.format(precision=2))
    # Show average yield
    avg_yield = np.mean(history_df["Prediction"])
    st.metric(label=T["average_yield"][lang], value=f"{avg_yield:.2f} g/m²")
st.markdown("</div>", unsafe_allow_html=True)