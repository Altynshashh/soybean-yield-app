
# 🌱 Soybean Yield Prediction Web App

This project is a web-based deep learning application developed with Streamlit for predicting soybean seed yield. It is designed for agronomists, researchers, and agriculture-focused companies to estimate soybean productivity based on physiological plant features. The app also includes a visual analysis of historical yield trends based on climate data.

---

## 🧠 Main Features

### 1. Physiological Yield Predictor (`app.py`)
- Input data manually or upload a CSV with soybean plant features.
- Predict yield in real time using a trained deep learning model (`.h5`).
- Visualizations include predicted yield distributions and key statistics.
- Supports English/Korean UI and downloadable results.
- Provides warnings and tips if the predicted yield is unusually low/high.

### 2. Climate Yield Viewer (`upd.py`)
- Visualize historical soybean yield across countries from 1960 to 2020.
- Analyze correlations between rainfall, average temperature, and yield.
- Supports scatterplot animations and data filtering by country/year.
- Feature marked as "under development" for future forecasting.

---

## 📁 Project Structure

```
├── app.py                             # Main Streamlit app (plant-based yield prediction)
├── upd.py                             # Historical climate yield visualization
├── soybean_yield_model.h5             # Pre-trained Keras model
├── Advanced Soybean Agricultural Dataset.csv
├── yield_df_soybeans.csv              # Historical yield & climate dataset
├── requirements.txt                   # Python dependencies
├── README.md
└── docs/
    ├── manual test.pdf                # Sample manual input
    └── upload test.pdf                # Sample CSV input
```

---

## 🚀 How to Run

1. **Install dependencies**  
```bash
pip install -r requirements.txt
```

2. **Run the Streamlit app**  
```bash
streamlit run app.py
```

3. *(Optional)* Launch climate analysis tool:  
```bash
streamlit run upd.py
```

---

## 📌 Future Plans

- Integrate real-time weather API for predictive modeling.
- Add satellite image or drone-based phenotyping input.
- Improve model explainability (e.g., SHAP/XAI).

---

© 2025 by Altynshash, for AI Convergence Graduate Program @ Chonnam National University.
