# 🏠 California Housing Price Prediction & Forecasting Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BenjaminBanansos/housing-forecast-ml/blob/main/Housing_forecast.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717.svg?logo=github)](https://github.com/BenjaminBanansos/housing-forecast-ml)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)

An applied machine learning and data engineering project built in **Google Colab** that analyzes California demographic and geographic census data to forecast median residential home values (`median_house_value`).

---

## 📌 Executive Summary

Predicting housing prices requires transforming highly skewed real-world distributions and accounting for multi-collinear geographic factors (such as proximity to the ocean and room-to-bedroom densities). This project implements an end-to-end predictive pipeline:

1. **Exploratory Data Analysis (EDA):** Discovered distribution skews and correlation patterns between geographic coordinates, income brackets, and property values.
2. **Feature Engineering:** Log-normal transformations, ratio formulations (`bedroom_ratio = total_bedrooms / total_rooms`), and one-hot encoding for categorical ocean proximity markers.
3. **Data Preprocessing:** Handled missing values and scaled feature matrices using `StandardScaler`.
4. **Machine Learning Benchmarking:** Trained and compared **Linear Regression** against an ensemble **Random Forest Regressor**.
5. **Automated Export:** Serialized test predictions to CSV (`prediction_table.csv`) for downstream analytics and business intelligence consumption.

---

## 🔬 Key Engineering & Modeling Steps

### 1. Data Cleaning & Statistical Transformation
* **Missing Value Handling:** Cleaned incomplete survey rows in attributes like `total_bedrooms`.
* **Log-Normal Transformation:** Raw room counts exhibited significant right-skew. Applied `np.log(total_rooms + 1)` to stabilize variance and normalize distributions for linear model convergence.
* **Domain Feature Engineering:** Engineered custom interaction terms including `bedroom_ratio` to capture housing density and luxury tiers.

### 2. Categorical Encoding
* Transformed categorical feature `ocean_proximity` (`<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`) via `pd.get_dummies` into binary indicator variables.

### 3. Model Training & Evaluation
* **Linear Regression:** Baseline estimator to gauge linear correlation between household income, geographic location, and home values.
* **Random Forest Regressor:** Non-linear ensemble model capturing complex non-linear geographic interactions, mitigating overfitting through bagging decision trees.
* **Evaluation:** Scored models using coefficient of determination ($R^2$) and residual error analysis.

---

## 🚀 How to Run the Project

### 1-Click Launch via Google Colab (Recommended)
Click the badge below to run the notebook interactively in your browser with zero local setup:

👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BenjaminBanansos/housing-forecast-ml/blob/main/Housing_forecast.ipynb)

---

## 📦 Tech Stack & Dependencies

* **Language:** Python 3.9+
* **Data Processing & Math:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (`LinearRegression`, `RandomForestRegressor`, `StandardScaler`, `train_test_split`)
* **Visualization:** `matplotlib`, `seaborn`, `plotly.express`
* **Interactive Environment:** Google Colab / Jupyter Notebook

---

## 👤 Author

**Benjamin Banansos**  
*Solutions Architect & Technical Product Owner*  
* Toronto, ON, Canada  
* **Email:** [benjaminbanansos@gmail.com](mailto:benjaminbanansos@gmail.com)  
* **Interactive Portfolio:** [https://benjaminbanansos.github.io/portfolio/](https://benjaminbanansos.github.io/portfolio/)  
* **GitHub:** [@BenjaminBanansos](https://github.com/BenjaminBanansos)
