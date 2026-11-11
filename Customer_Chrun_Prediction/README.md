# Customer Churn Prediction Using Machine Learning

## Project Overview

Customer Churn Prediction is a Machine Learning project designed to identify customers who are likely to discontinue using a company's products or services.

Customer retention is important for businesses because acquiring new customers can be more expensive than retaining existing ones. By analyzing customer demographics, service subscriptions, contract details, billing information, and customer behavior, this project aims to predict whether a customer is likely to leave the company.

This project implements multiple classification algorithms, compares their performance using evaluation metrics, and includes a Streamlit application for interactive customer churn prediction.

## Problem Statement

Customer churn is one of the major challenges faced by subscription-based businesses, particularly telecommunications companies.

When customers discontinue their subscriptions, businesses may experience revenue loss and increased customer acquisition costs.

The objective of this project is to develop a Machine Learning classification model that analyzes historical customer information and predicts whether a customer is likely to churn.

## Project Objectives

- Analyze customer information and identify potential churn patterns.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Convert categorical variables into numerical representations.
- Apply feature scaling to numerical features.
- Train multiple Machine Learning classification algorithms.
- Compare models using different evaluation metrics.
- Identify customers who are more likely to discontinue services.
- Save trained Machine Learning models for future predictions.
- Develop an interactive application using Streamlit.

## Dataset Information

The project uses a telecommunications customer churn dataset.

**Dataset Details:**

| Attribute | Description |
|---|---|
| Total Records | 7,043 |
| Total Features | 21 |
| Target Variable | Churn |
| Problem Type | Binary Classification |
| Missing Values | 11 in TotalCharges |
| Data Format | CSV |

### Dataset Features

The dataset contains the following customer information:

| Feature | Description |
|---|---|
| customerID | Unique customer identification |
| gender | Customer gender |
| SeniorCitizen | Whether the customer is a senior citizen |
| Partner | Whether the customer has a partner |
| Dependents | Whether the customer has dependents |
| tenure | Number of months the customer has stayed |
| PhoneService | Whether phone service is subscribed |
| MultipleLines | Multiple phone line subscription details |
| InternetService | Internet service type |
| OnlineSecurity | Online security subscription |
| OnlineBackup | Online backup subscription |
| DeviceProtection | Device protection subscription |
| TechSupport | Technical support subscription |
| StreamingTV | Streaming TV subscription |
| StreamingMovies | Streaming movies subscription |
| Contract | Customer contract type |
| PaperlessBilling | Whether paperless billing is enabled |
| PaymentMethod | Customer payment method |
| MonthlyCharges | Monthly subscription charges |
| TotalCharges | Total customer charges |
| Churn | Whether the customer discontinued the service |

**Target Variable:**

- `0` – Customer did not churn.
- `1` – Customer churned.

## Technologies & Tools Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| Scikit-learn | Machine Learning algorithms and evaluation |
| XGBoost | Gradient boosting classification |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web application |
| Jupyter Notebook | Development environment |
| Git & GitHub | Version control and project hosting |

## Project Workflow

### 1. Data Collection

- Imported the customer churn dataset using Pandas.
- Loaded the CSV file into a DataFrame.
- Examined the dataset using `head()` and `describe()`.
- Inspected the dataset structure and data types using `info()`.

### 2. Data Cleaning

Performed the following preprocessing operations:

- Identified missing values in the `TotalCharges` column.
- Removed 11 records containing missing `TotalCharges` values.
- Converted `TotalCharges` into a numerical data type.
- Converted `SeniorCitizen` into a categorical variable.
- Removed `customerID` because it is a unique identifier and was not required for model training.

### 3. Data Preprocessing

Converted categorical variables into numerical representations.

**Binary Encoding:**

Converted Yes/No categorical values into numerical values.

- Yes → 1
- No → 0

Applied binary encoding to:
- Partner
- Dependents
- PhoneService
- PaperlessBilling
- Churn

Gender encoding:
- Male → 1
- Female → 0

**One-Hot Encoding:**

Applied one-hot encoding to categorical features containing multiple categories using Pandas `get_dummies()`.

Features included:
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaymentMethod

### 4. Feature Scaling

Applied `StandardScaler` to numerical features to standardize their values.

Features selected for scaling:

- tenure
- MonthlyCharges
- TotalCharges

### 5. Train-Test Split

Divided the dataset into training and testing datasets.

| Parameter | Value |
|---|---|
| Training Data | 80% |
| Testing Data | 20% |
| Random State | 42 |
| Stratification | Enabled |
| Training Records | 5,625 |
| Testing Records | 1,407 |

Used `train_test_split()` with stratification to maintain the target class distribution.

### 6. Machine Learning Model Development

Implemented and compared five classification algorithms.

**Algorithms Used:**

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

Each model was trained using the training dataset and evaluated using the testing dataset.

### 7. Model Evaluation

Evaluated model performance using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

Also generated:

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

These evaluation techniques help compare classification performance and understand how effectively the models identify customers who are likely to churn.

## Model Performance Comparison

The following results were obtained from the notebook's model evaluation.

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.38% | 64.85% | 57.22% | 60.80% | 83.59% |
| Random Forest | 79.39% | 63.73% | 52.14% | 57.35% | 81.74% |
| SVM | 78.96% | 63.27% | 49.73% | 55.69% | 77.93% |
| KNN | 75.84% | 54.52% | 54.81% | 54.67% | 77.78% |
| XGBoost | 76.83% | 57.10% | 51.60% | 54.21% | 81.40% |

**Model Evaluation Observation:**

Logistic Regression achieved the best overall results among the five evaluated algorithms:

- Accuracy: 80.38%
- Precision: 64.85%
- Recall: 57.22%
- F1-Score: 60.80%
- ROC-AUC: 83.59%

Based on the reported evaluation metrics, Logistic Regression demonstrated the strongest performance in this experiment.

**Note:** The current notebook saves the Random Forest model as `churn_model.pkl`. The saved model should be updated to Logistic Regression if the intention is to deploy the best-performing model from this comparison.

## Model Visualization

The project includes visual evaluation techniques:

**1. Confusion Matrix**

Used to understand correct and incorrect customer churn predictions.

**2. ROC Curve**

Used to evaluate the model's ability to distinguish between churn and non-churn customers at different classification thresholds.

**3. Precision-Recall Curve**

Used to understand the relationship between precision and recall when identifying customers who may churn.

## Model Saving and Loading

Used Joblib to save the trained Machine Learning model.

Model artifact:

`churn_model.pkl`

These files support loading the trained model and maintaining the feature preprocessing structure for future predictions.

## Streamlit Web Application

Developed an interactive customer churn prediction application using Streamlit.

The application collects customer information through interactive input fields.

**Input Features Include:**

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

**Application Functionality:**

1. Collect customer information.
2. Preprocess customer input.
3. Load the trained Machine Learning model.
4. Generate a churn prediction.
5. Display the prediction result.
6. Display the churn probability.

**Prediction Output:**

- Churn – Customer is predicted to discontinue the service.
- No Churn – Customer is predicted to continue the service.

The application also displays the predicted churn probability.

## Installation and Setup

### Prerequisites

Make sure Python and pip are installed on your system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/SanthanamariG/Machine-Learning-Projects.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Customer-Churn-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

Windows:

```powershell
venv\Scripts\activate
```

Linux:

```bash
source venv/bin/activate
```

### Step 5: Run the Streamlit Application

```bash
streamlit run src/app.py
```

## Requirements

The project uses the following Python libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
streamlit
jupyter
```

## Future Improvements

- Improve model performance through hyperparameter tuning.
- Implement cross-validation for more reliable model evaluation.
- Address class imbalance using appropriate techniques.
- Perform feature importance analysis.
- Optimize prediction performance.
- Improve the Streamlit application's user interface.
- Deploy the application using a cloud platform.
- Integrate automated model training and deployment pipelines.

## Key Learnings

Through this project, I gained practical experience in:

- Working with real-world customer datasets.
- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- Feature encoding and feature scaling.
- Implementing classification algorithms.
- Comparing Machine Learning models.
- Evaluating models using classification metrics.
- Saving and loading trained models.
- Building interactive applications using Streamlit.
