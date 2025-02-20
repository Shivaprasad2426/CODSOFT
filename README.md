# CODSOFT Data Science Internship Projects

This repository contains my work for the **CodSoft Data Science Internship**. The projects involve data analysis, machine learning, and predictive modeling.

## 📌 Projects Included:
1. **Titanic Survival Prediction** – Classification model predicting passenger survival.
2. **Sales Prediction** – Forecasting sales using machine learning.
3. **Credit Card Fraud Detection** – Identifying fraudulent transactions.

## 📂 Repository Structure:
CODSOFT/ ├── Titanic_Survival/ │ ├── titanic_model.ipynb │ ├── titanic_data.csv │ ├── README.md  ├── README.md ├── Sales_Prediction/ │ ├── sales_model.ipynb │ ├── sales_data.csv │ ├── README.md ├── Credit_Card_Fraud_Detection/ │ ├── fraud_model.ipynb │ ├── creditcard_data.csv │ ├── README.md

## 📜 How to Run:
1. Clone the repository:  https://github.com/Shivaprasad2426/CODSOFT.git

# 🚢 Titanic Survival Prediction

## 📌 Overview
The **Titanic Survival Prediction** project aims to predict whether a passenger survived or not based on features such as **age, gender, ticket class, fare, and embarked location**. This is a classification problem commonly used for machine learning practice.

## 📂 Dataset
- **Source:** [Kaggle - Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **Features Included:**
  - `PassengerId`: Unique ID for each passenger.
  - `Pclass`: Ticket class (1st, 2nd, or 3rd).
  - `Sex`: Gender of the passenger.
  - `Age`: Age of the passenger.
  - `SibSp`: Number of siblings/spouses aboard.
  - `Parch`: Number of parents/children aboard.
  - `Fare`: Fare paid for the ticket.
  - `Embarked`: Port of embarkation (C, Q, S).
- **Target Variable:**
  - `Survived`: 0 = No, 1 = Yes.

## 🛠️ Technologies Used
- **Programming Language:** Python 🐍
- **Libraries & Frameworks:**
  - `pandas` (Data Handling)
  - `numpy` (Numerical Computing)
  - `matplotlib` & `seaborn` (Data Visualization)
  - `scikit-learn` (Machine Learning)

## 📊 Exploratory Data Analysis (EDA)
- Checked for missing values and handled them.
- Visualized survival rates based on ticket class, gender, and age.
- Created correlation heatmaps to find relationships between features.

### 📈 **Key Insights from EDA**
- **Females** had a higher survival rate.
- **First-class passengers** had a better chance of survival.
- **Younger passengers** had higher survival rates.

## 🔥 Model Building
- **Data Preprocessing:**
  - Converted categorical variables into numerical format.
  - Scaled numerical features using `StandardScaler`.
  - Handled missing values in **Age** and **Embarked** columns.
  
- **Trained Machine Learning Models:**
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - XGBoost Classifier
  
- **Hyperparameter Tuning**:
  - Used `GridSearchCV` to find the best hyperparameters for Random Forest and SVM.

## 🎯 Results & Model Evaluation
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|-----------|
| Logistic Regression | 80.4% | 78.2% | 72.5% | 75.3% |
| Random Forest       | 85.6% | 83.1% | 78.7% | 80.8% |
| SVM                 | 81.2% | 79.4% | 74.6% | 76.9% |
| XGBoost             | **87.2%** | **85.0%** | **79.3%** | **81.9%** |

✅ **Best Model:** **XGBoost Classifier** (87.2% Accuracy)

## 📜 How to Run the Project
1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/CODSOFT.git
   cd Titanic_Survival





# 📊 Sales Prediction

## 📌 Project Overview
This project forecasts sales using historical data and regression techniques.

## 📂 Dataset
- [Sales Prediction Dataset](https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/input)

## 🚀 Technologies Used
- Python, Pandas, NumPy
- Scikit-learn (Linear Regression, Random Forest)
- Matplotlib & Seaborn

## 📜 How to Run
1. Open `sales_model.ipynb` in Jupyter Notebook.
2. Run all cells to preprocess data, train the model, and visualize predictions.


# 💳 Credit Card Fraud Detection

## 📌 Project Overview
This project builds a classification model to detect fraudulent credit card transactions.

## 📂 Dataset
- [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🚀 Technologies Used
- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest, SMOTE for imbalance handling)
- Matplotlib & Seaborn

## 📜 How to Run
1. Open `fraud_model.ipynb` in Jupyter Notebook.
2. Run all cells to preprocess data, train the model, and evaluate fraud detection performance.

