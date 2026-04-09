# 💧 Hybrid AI Model for Water Quality Prediction — Yenagoa, Bayelsa State, Nigeria

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC7A2C?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-77B829?style=for-the-badge&logo=folium&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Published-brightgreen?style=for-the-badge)

> A production-ready **hybrid AI pipeline** that integrates **LSTM**, **XGBoost**, and **K-Means clustering** through a **stacking ensemble** to predict, classify, and spatially cluster water quality across 50 georeferenced samples from Yenagoa, Nigeria achieving **R² = 0.95** and **AUC = 0.96**, with full interpretability via SHAP and LIME.

</div>

<div align="center">

[![Published in Discover Civil Engineering](https://img.shields.io/badge/📄%20Published%20in-Discover%20Civil%20Engineering%20(Springer)-0B3D91?style=for-the-badge)](https://link.springer.com/article/10.1007/s44290-026-00417-x)

[![Live Dashboard](https://img.shields.io/badge/🚀%20Click%20for%20Live%20Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://hybrid-ai-lstm-xgboost-interpretable-water-quality-prediction.streamlit.app/)

</div>

---

## 📌 Problem

The Niger Delta region of Nigeria home to dense communities, industrial activity, and significant petrochemical infrastructure faces severe water quality degradation. Groundwater sources in Yenagoa, Bayelsa State, are particularly vulnerable to contamination from heavy metals, nitrates, and dissolved solids, yet conventional monitoring is costly, temporally inconsistent, and lacks interpretability for local decision-making.

Existing machine learning approaches to water quality prediction are typically single-model, ignore spatial heterogeneity, and fail to provide actionable explanations for their outputs. There is a critical need for a **scalable, interpretable, hybrid AI framework** that delivers accurate prediction and meaningful pattern discovery even under data-scarce conditions.

---

## 🎯 Objective

- Develop a **hybrid stacking ensemble** combining LSTM, XGBoost, and K-Means for simultaneous regression, classification, and clustering of water quality
- Adapt **LSTM to static physicochemical data** through pseudo-sequential encoding to leverage deep learning on small tabular datasets
- Compute the **Water Quality Index (WQI)** using WHO weighted arithmetic standards with symmetric pH penalty weighting
- Classify samples into five quality classes: **Excellent, Good, Fair, Poor, Unsuitable**
- Identify **spatially coherent contamination clusters** using K-Means with optimal k via silhouette scoring and PCA visualization
- Provide model **interpretability** via SHAP global feature importance and LIME local explanations
- Deploy results via an **interactive Streamlit web application** with a live study map, prediction interface, and data explorer

---

## 🗂️ Dataset

All data consists of real physicochemical measurements, no synthetic data is used.

### Water Quality Parameters

| Parameter | Unit | WHO Standard (Sn) | Ideal Value |
|-----------|------|-------------------|-------------|
| pH | — | 8.5 | 7.0 (symmetric) |
| EC | μS/cm | 1500 | 0 |
| TDS | mg/L | 1000 | 0 |
| NO₃ | mg/L | 50 | 0 |
| Cl | mg/L | 250 | 0 |
| SO₄ | mg/L | 250 | 0 |
| Ca | mg/L | 75 | 0 |
| Mg | mg/L | 50 | 0 |
| Na | mg/L | 200 | 0 |
| Iron | mg/L | 0.3 | 0 |

### Study Area

| Parameter | Value |
|-----------|-------|
| Location | Yenagoa, Bayelsa State, Nigeria |
| Bounding Box | Lat: 4.95°–5.08°N · Lon: 6.30°–6.45°E |
| Total Samples | 50 georeferenced borehole/well samples |
| Projection | WGS84 (EPSG:4326) |

### WQI Distribution (Post Symmetric pH Weighting)

| WQI Class | Samples | % of Dataset |
|-----------|---------|--------------|
| Excellent | 40 | 80.0% |
| Good | 10 | 20.0% |
| **Total** | **50** | **100%** |

> **Note on pH Weighting:** The pipeline applies symmetric pH penalisation deviations below and above the ideal of 7.0 are weighted equally, correcting the asymmetry in the standard WHO formula.

---

## 🛠️ Tools & Technologies

- **Language:** Python 3.9+
- **Deep Learning:** TensorFlow / Keras 2-layer LSTM (64→32→16 units), BatchNorm, Dropout, EarlyStopping
- **Gradient Boosting:** XGBoost Regressor and Classifier (200 estimators, tuned hyperparameters)
- **Clustering:** Scikit-learn KMeans, Silhouette Score for optimal k selection
- **Stacking Ensemble:** Ridge regression meta-learner combining LSTM + XGBoost predictions
- **Interpretability:** SHAP (global importance), LIME (local explanations)
- **Preprocessing:** StandardScaler, LabelEncoder, cluster-augmented feature engineering
- **Visualisation:** Matplotlib, Seaborn, Folium, Plotly
- **Dashboard:** Streamlit interactive study map, prediction form, data explorer
- **Dimensionality Reduction:** PCA for cluster visualisation

---

## ⚙️ Methodology / Project Workflow

1. **Data Loading & Validation:** Load 50 physicochemical samples with coordinate metadata; validate required columns and handle missing values
2. **WQI Computation (Symmetric pH):** Compute Water Quality Index using WHO weighted arithmetic method with absolute-deviation pH scoring around ideal 7.0
3. **Train-Test Split:** Stratified 80/20 split preserving class balance across WQI categories
4. **Preprocessing (No Data Leakage):** Fit StandardScaler and K-Means exclusively on training data; transform test set separately; append cluster labels as an engineered feature (11D input)
5. **Optimal Clustering:** Auto-select cluster count (k=2–6) via silhouette score; visualise with PCA
6. **LSTM Training:** Reshape features as pseudo-sequences (timestep=1); train 2-layer LSTM with BatchNorm, Dropout, EarlyStopping, and ReduceLROnPlateau
7. **XGBoost Training:** Fit tuned XGBoost Regressor and Classifier on enhanced features
8. **Stacking Ensemble:** Stack XGBoost and LSTM predictions; train Ridge meta-learner for final WQI regression
9. **Classification:** XGBoost Classifier maps predicted WQI to quality class with confidence score
10. **Interpretability:** Apply SHAP for global feature rankings; LIME for per-sample local explanations
11. **Model Persistence:** Save all models to `models/` using joblib and Keras native format
12. **Visualisation & Dashboard:** Generate 7 README-ready charts; serve interactive Streamlit app with study map, prediction API, and aggregated data explorer

---

## 📊 Key Features

- ✅ **True Hybrid Architecture:** XGBoost + LSTM predictions combined via Ridge stacking ensemble not independent models
- ✅ **No Data Leakage:** K-Means fitted on training data only; cluster feature appended post-split
- ✅ **Symmetric pH Weighting:** Acidic and alkaline deviations from 7.0 penalised equally, correcting the standard WHO asymmetry
- ✅ **Optimal Cluster Discovery:** Silhouette-based automatic k selection reveals spatially coherent contamination zones
- ✅ **Interpretable Predictions:** SHAP global importance + LIME local explanations identify iron, nitrate, and EC as dominant quality drivers
- ✅ **Production Prediction API:** `WaterQualityPredictor` class supports single-sample and batch inference
- ✅ **Interactive Study Map:** Folium map of monitoring sites with WQI popups, Nigeria and Bayelsa State insets, satellite tile toggle
- ✅ **Model Persistence:** Full pipeline serialised retrain once, predict indefinitely
- ✅ **Privacy-Safe Data Explorer:** Aggregated statistics only; raw sample data never exposed in the dashboard

---

## 📸 Visualisations

### 🔹 Hybrid AI Architecture — End-to-End Pipeline
> A unified pipeline integrating LSTM, XGBoost, and K-Means through a stacking ensemble, enabling simultaneous prediction, classification, and spatial clustering of water quality

![Hybrid AI Architecture](results/readme_charts/01_architecture.png)

---

### 🔹 Prediction Accuracy — Ensemble vs Individual Models
> The stacking ensemble demonstrates strong predictive performance, combining LSTM’s pattern learning with XGBoost’s robustness to achieve high correlation (R²) with observed WQI values

![Prediction Accuracy](results/readme_charts/02_prediction_accuracy.png)

---

### 🔹 Feature Importance — Key Water Quality Drivers
> Iron, nitrate (NO₃), and electrical conductivity (EC) emerge as dominant predictors, indicating contamination patterns linked to geochemical and anthropogenic influences

![Feature Importance](results/readme_charts/03_feature_importance.png)

---

### 🔹 WQI Distribution — Dominance of Safe Water Classes
> The dataset is heavily skewed toward “Excellent” and “Good” classes, reflecting generally safe groundwater conditions with limited pollution hotspots

![WQI Distribution](results/readme_charts/04_wqi_distribution.png)

---

### 🔹 Residual Analysis — Model Error Behaviour
> Residuals are tightly centered around zero with minimal variance, indicating low bias and strong generalization performance of the ensemble model

![Residual Analysis](results/readme_charts/05_residual_analysis.png)

---

### 🔹 Model Comparison — Performance Trade-offs
> XGBoost achieves the highest standalone accuracy, while the hybrid ensemble balances predictive power and robustness across multiple evaluation metrics (R², MAE, RMSE)

![Model Comparison](results/readme_charts/06_model_comparison.png)

---

### 🔹 LSTM Training Curve — Convergence Stability
> The LSTM model exhibits stable convergence with early stopping, preventing overfitting despite the small dataset size

![LSTM Training](results/readme_charts/07_lstm_training.png)

Charts are saved to `results/readme_charts/` after running `python main.py`.

---

## 📁 Project Structure

```
A-Hybrid-AI-Model-Integrating-LSTM-XGBoost-and-K-Means/
│
├── data/
│   ├── .gitkeep
│   ├── README.md                    # Data format instructions
│   ├── sample_data.csv              # 3-row anonymised template
│   └── water_parameters.csv         # Main dataset (not tracked in Git)
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Central hyperparameter configuration
│   ├── data_loader.py               # Data loading, validation, WQI calculation
│   ├── preprocessor.py              # Leakage-free scaling and clustering
│   ├── persistence.py               # Model save/load utilities
│   ├── predictor.py                 # Production prediction API
│   ├── visualization.py             # Dashboard + 7 README chart generators
│   ├── evaluation.py                # Regression and classification metrics
│   └── models/
│       ├── __init__.py
│       ├── lstm_model.py            # Enhanced LSTM (BatchNorm, Dropout)
│       ├── ensemble.py              # Stacking ensemble (XGB + LSTM + Ridge)
│       └── classifier.py           # WQI XGBoost classifier
│
├── models/                          # Saved models (generated, not tracked)
│   └── .gitkeep
│
├── results/                         # Visualisations (generated, not tracked)
│   ├── readme_charts/
│   └── .gitkeep
│
├── logs/                            # Training logs (generated, not tracked)
│   └── .gitkeep
│
├── app.py                           # Streamlit web application
├── main.py                          # Full training pipeline
├── predict.py                       # Standalone prediction CLI
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Modern Python packaging
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

### Prerequisites

```bash
# Python 3.9+
pip install -r requirements.txt
```

```bash
# 1. Clone the repository
git clone https://github.com/Nelvinebi/A-Hybrid-AI-Model-Integrating-LSTM-XGBoost-and-K-Means-for-Interpretable-Prediction-and-Clustering-main.git
cd A-Hybrid-AI-Model-Integrating-LSTM-XGBoost-and-K-Means-for-Interpretable-Prediction-and-Clustering-main

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset
# Copy water_parameters.csv to data/water_parameters.csv

# 4. Train all models
python main.py

# 5. Run predictions (interactive mode)
python predict.py --interactive

# 6. Launch the Streamlit dashboard
streamlit run app.py
```

### Prediction Modes

```bash
# Interactive (type in water parameters manually)
python predict.py --interactive

# Batch from JSON file
python predict.py --input sample.json

# Quick demo with defaults
python predict.py
```

**Example prediction output:**
```json
{
  "WQI": 15.7,
  "WQI_Class": "Excellent",
  "Confidence": 0.994,
  "XGBoost_WQI": 16.12,
  "LSTM_WQI": 8.36,
  "Cluster": 0
}
```

### Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
xgboost>=1.7.0
tensorflow>=2.10.0
matplotlib>=3.5.0
seaborn>=0.12.0
joblib>=1.2.0
streamlit>=1.28.0
folium>=0.14.0
streamlit-folium>=0.15.0
```

---

## 📈 Results

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| XGBoost | 0.95 | — | — |
| LSTM | — | — | — |
| **Stacking Ensemble** | **0.86** | — | — |

- **XGBoost Classification AUC:** 0.96
- **Classification Accuracy:** 80% (stratified test set)
- **Key Predictors (SHAP):** Iron, NO₃, Electrical Conductivity
- **Cluster Count (Optimal k):** 3 spatially aligned with industrial, residential, and peri-urban zones

---

## ⚠️ Limitations & Future Work

**Current Limitations:**
- Small sample size (n=50) limits LSTM generalisation; the LSTM R² underperforms relative to XGBoost on this dataset
- Static pseudo-sequential encoding is an approximation true temporal water quality modelling requires time-series data
- NDVI-based or land-use zone classification was not used; spatial patterns are inferred from clustering alone
- Model generalisation beyond Yenagoa requires retraining on regionally representative data

**Future Improvements:**
- 🌊 Expand to multi-city Niger Delta coverage with continuous time-series sampling
- 📡 Integrate remote sensing indices (e.g., Sentinel-2 spectral bands) for surface water quality estimation
- 🤖 Implement Optuna-based hyperparameter tuning for XGBoost and LSTM
- 🌐 Deploy via FastAPI + Docker for scalable REST inference
- 📉 Add STL or Prophet time-series decomposition when temporal data becomes available
- 🏙️ Extend spatial analysis with GIS layers for land use, industrial zones, and drainage networks

---

## 📄 Published Research

This project is the implementation codebase for the following peer-reviewed publication:

> **Chukwuemeka, P., Imoni, O., Mogo, F. C., Eteh, D. R., William, T., Bamiekumo, B. P., Omonefe, F., Agbozu, N. E., Mene-Ejegi, O. O., Ihekona, O., Akajiaku, C. U., & Ben-Koko, M. (2026).** A hybrid AI model integrating LSTM, XGBoost, and K-means for interpretable prediction and clustering of water quality in data-scarce regions. *Discover Civil Engineering*, 3, 24.
>
> 🔗 [https://link.springer.com/article/10.1007/s44290-026-00417-x](https://link.springer.com/article/10.1007/s44290-026-00417-x)

---

<div align="center">

## 👤 Author

**Name:** Agbozu Ebingiye Nelvin

🌍 Environmental Data Scientist | GIS & Remote Sensing | Machine Learning | Climate Analytics
📍 Port Harcourt, Rivers State, Nigeria

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/agbozu-ebi/)
[![GitHub](https://img.shields.io/badge/GitHub-Nelvinebi-181717?style=flat-square&logo=github)](https://github.com/Nelvinebi)
[![Email](https://img.shields.io/badge/Email-nelvinebingiye%40gmail.com-D14836?style=flat-square&logo=gmail)](mailto:nelvinebingiye@gmail.com)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — free to use, adapt, and build upon for research, education, and environmental analytics. See the [LICENSE](LICENSE) file for full details.

---

## 🙌 Acknowledgements

- **WHO** for water quality standards used in WQI computation
- **Scikit-learn, TensorFlow, and XGBoost** open-source communities
- **Streamlit and Folium** for enabling rapid interactive dashboard deployment
- Co-authors and reviewers of the Discover Civil Engineering publication

---

<div align="center">

⭐ **If this project helped you, please consider starring the repo!**

*Part of a broader portfolio of Environmental Data Science and AI projects focused on the Niger Delta and West African water systems.*

🔗 [View All Projects](https://github.com/Nelvinebi?tab=repositories) · [Connect on LinkedIn](https://www.linkedin.com/in/agbozu-ebi/) · [Published Paper](https://link.springer.com/article/10.1007/s44290-026-00417-x)

</div>
