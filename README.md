# Customer_Churn_Prediction-

# 📊 Customer Churn Prediction App

A machine learning web application that predicts whether a customer is likely to churn, built with **Python**, **Scikit-learn**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly)

---

## 🚀 Live Demo

> Add your Streamlit Cloud link here after deploying

---

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **99.95%** |
| ROC-AUC | **0.997** |
| Precision | High |
| Recall | High |

---

## 🧠 How It Works

1. User inputs customer details (age, tenure, contract type, etc.)
2. Input is scaled using a pre-trained `StandardScaler`
3. Scaled input is passed to a **Random Forest Classifier**
4. App displays churn probability with visual charts

---

## 📊 Features

- ✅ Real-time churn prediction
- 📈 Interactive **Gauge chart** showing risk level
- 🍩 **Donut chart** showing churn vs retention split
- 🟢🟡🔴 Risk level indicator (Low / Moderate / High)
- 📋 Input summary table
- 🎛️ Clean 4-column input layout

---

## 🔍 Input Features

| Feature | Type |
|--------|------|
| Age | Numeric |
| Tenure (Months) | Numeric |
| Monthly Charges | Numeric |
| Total Charges | Numeric |
| Gender | Categorical |
| Contract Type | Categorical |
| Internet Service | Categorical |
| Tech Support | Categorical |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Scikit-learn | ML model + scaler |
| Streamlit | Web app UI |
| Pandas | Data manipulation |
| Plotly | Interactive charts |
| Joblib | Model serialization |

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run App.py
```

---

## 📁 Project Structure

```
Customer_Churn/
│
├── App.py                      # Streamlit web app
├── Customer.ipynb              # EDA + model training notebook
├── scaler.pkl                  # Saved StandardScaler
├── Customer_churn_model.pkl    # Saved Random Forest model
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## 📦 Requirements

```
streamlit
pandas
scikit-learn
joblib
plotly
numpy
```

---

## 📌 Model Training Summary

- Dataset: 4M+ customer records
- Algorithm: Random Forest Classifier
- Preprocessing: StandardScaler, One-Hot Encoding
- Evaluation: Accuracy, ROC-AUC, Precision, Recall

---

## 👨‍💻 Author

**Bhushan Gavali**
- 🔗 [LinkedIn](https://linkedin.com/in/bhushangavali)
- 🐙 [GitHub](https://github.com/bhushangavali)
- 📧 bhushangavali24@gmail.com

---

## ⭐ If you found this useful, please star the repo!
