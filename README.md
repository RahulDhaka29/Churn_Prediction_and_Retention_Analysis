# 📉 Customer Churn Prediction & Retention Analysis

## 🧩 Overview
This project tackles the **critical business problem of customer churn** within the telecommunications industry.  
Using a dataset of Telco customer information, it aims to understand **why customers leave (retention analysis)** and build a **machine learning model** to predict customers at risk of churning.  
The project culminates in an **interactive Streamlit web application** that showcases the analysis and enables live churn predictions.

---

## 🎯 Goals

- **Retention Analysis:** Identify the key factors and customer segments associated with higher churn rates.  
- **Churn Prediction:** Develop a robust binary classification model optimized for **Recall** to effectively identify customers at high risk of churning.  
- **Deployment:** Build an interactive **Streamlit dashboard** to present findings and provide a live prediction tool.

---

## 📊 Dataset

- **Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Description:** Contains information about ~7,000 customers, including demographics, subscribed services, account details (tenure, contract, payment method, charges), and churn labels.  
- **File Used:** `Telco-Customer-Churn.csv`

---

## 📁 Project Structure

```
.
├── Telco-Customer-Churn.csv          # The raw dataset
├── churn_analysis.ipynb              # Jupyter Notebook with full analysis and model training
├── preprocessor.pkl                  # Saved scikit-learn preprocessor pipeline
├── model.pkl                         # Saved trained machine learning model
├── model_comparison_results.csv      # Model performance metrics across algorithms
├── app.py                            # Streamlit application for UI and live predictions
└── README.md                         # Project documentation
```

---

## ⚙️ Methodology

### 1. Data Loading & Exploration
- Loaded the dataset using **Pandas**
- Checked data types (`.info()`), duplicates (`.duplicated()`), and statistical summaries (`.describe()`)

### 2. Data Cleaning & Preprocessing
- Converted `TotalCharges` from object → numeric, filled missing values (0 for zero-tenure customers)
- Dropped irrelevant `customerID` column
- Encoded target variable `Churn`: `'Yes' → 1`, `'No' → 0`

### 3. Exploratory Data Analysis (EDA) & Retention Analysis
- Visualized overall churn rate
- Analyzed churn by categorical features using **Plotly histograms**
- Analyzed numerical features (`tenure`, `charges`) using **violin plots**
- Correlation analysis using **Seaborn heatmap**

### 4. Feature Engineering & Preprocessing
- Split dataset into features (`X`) and target (`y`)
- Used **train-test split (80/20)**, stratified by churn
- Applied **ColumnTransformer**:
  - `StandardScaler` → numerical features  
  - `OneHotEncoder` → categorical features
- Saved fitted preprocessor as `preprocessor.pkl`

### 5. Model Tuning & Comparison
- Tested multiple classifiers:
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Classifier (SVC)  
  - Random Forest  
  - Gradient Boosting  
  - XGBoost
- Used **RandomizedSearchCV (5-fold CV)** optimizing for **Recall (Churn=1)**
- Evaluated models using Recall, Precision, F1-Score, Accuracy, and ROC AUC
- Selected the model with **highest Recall** and balanced Precision — typically **Logistic Regression**

### 6. Model Interpretation & Saving
- Extracted and visualized **feature importances** (coefficients)
- Confirmed findings with EDA insights
- Saved trained model as `model.pkl`

---

## 🔍 Key Findings & Insights

- **Overall Churn Rate:** ~26.5% of customers churned  
- **Top Churn Drivers:**
  - Month-to-month contracts  
  - Low tenure customers  
  - Fiber optic internet users  
  - Lack of OnlineSecurity / TechSupport  
  - Electronic check payment users  
- **Retention Factors:**
  - Long-term contracts (1–2 years)  
  - Longer tenure  
  - Bundled services like OnlineSecurity and TechSupport  

---

## 📈 Model Performance (Example Metrics)

| Metric | Value |
|:-------|:------|
| Recall (Churn=1) | ~0.76 |
| Precision (Churn=1) | ~0.53 |
| Accuracy | ~0.75 |
| ROC AUC | ~0.83 |

> *(Replace with actual values from `model_comparison_results.csv`)*

---

## 💻 Streamlit Web Application

The **interactive web app (`app.py`)** provides:
- **Project overview & insights**
- **Visual summaries** of churn analysis
- **Live Churn Prediction Tool**  
  → Enter customer details → Get churn probability and prediction instantly

### 🔧 Run the App

Make sure the following files are in the same directory:
```
app.py
preprocessor.pkl
model.pkl
Telco-Customer-Churn.csv
model_comparison_results.csv
```

Then run:

```bash
streamlit run app.py
```

---

## 🧰 Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Activate it
# On Windows:
.env\Scriptsctivate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> Generate `requirements.txt` using:
> ```bash
> pip freeze > requirements.txt
> ```

### 4. Run the Notebook
```bash
jupyter notebook churn_analysis.ipynb
```

---

## 📦 Essential Libraries

- `pandas`
- `numpy`
- `scikit-learn==1.7.2`
- `plotly`
- `seaborn`
- `matplotlib`
- `streamlit`
- `xgboost`

---

## 🧠 Future Enhancements

- Incorporate **Deep Learning** models (e.g., ANN)
- Add **SHAP** or **LIME** for advanced interpretability
- Include **real-time data ingestion** from APIs or databases
- Enhance UI/UX with better visuals and customer segmentation filters

---

## 🏁 Conclusion

This project demonstrates the full lifecycle of a **data-driven customer churn prediction system**, from data analysis and modeling to deployment.  
It helps telecom companies proactively **identify at-risk customers** and design effective **retention strategies** to reduce churn and boost loyalty.

---

**Developed by:** *[Rahul Dhaka]*  
**Course/Domain:** Machine Learning / Data Science  
**Tool Stack:** Python · Scikit-learn · Streamlit · Plotly · Pandas


## 📜 License

This project is open-source and available under the [MIT License](LICENSE).