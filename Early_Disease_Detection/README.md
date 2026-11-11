# Early Disease Detection Using Machine Learning

## Project Overview

Early Disease Detection is a Machine Learning project developed to predict the possibility of heart disease based on an individual's health indicators, medical measurements, and lifestyle information.

The project analyzes important health-related attributes such as age, gender, blood pressure, cholesterol levels, glucose levels, weight, height, smoking habits, alcohol consumption, and physical activity.

Multiple Machine Learning classification algorithms are implemented and evaluated to identify patterns in the dataset and predict the possibility of heart disease.

Additionally, an interactive web application is developed using Streamlit to allow users to enter health-related information and receive a prediction from the trained Machine Learning pipeline.

## Problem Statement

Heart disease is one of the major health challenges worldwide. Identifying potential risk indicators at an early stage can help individuals seek appropriate medical attention.

Traditional assessment methods require healthcare professionals to examine multiple health indicators.

The objective of this project is to develop a Machine Learning-based prediction system that analyzes health-related information and identifies potential heart disease risk patterns.

## Project Objectives

- Analyze health-related datasets to identify disease patterns.
- Perform data cleaning and preprocessing.
- Handle missing values and irrelevant features.
- Conduct Exploratory Data Analysis (EDA).
- Perform feature engineering and categorical encoding.
- Apply feature scaling to numerical variables.
- Implement multiple Machine Learning classification algorithms.
- Evaluate and compare model performance.
- Develop a prediction pipeline using Scikit-learn.
- Build an interactive prediction application using Streamlit.

## Dataset Information

The project uses a heart disease dataset containing health indicators, demographic information, and lifestyle-related attributes.

**Dataset Details:**

| Attribute | Description |
|---|---|
| Dataset Type | Health and Medical Data |
| Target Variable | disease |
| Problem Type | Binary Classification |
| Data Format | CSV |
| Prediction | Heart Disease / No Heart Disease |

### Dataset Features

| Feature | Description |
|---|---|
| id | Unique record identifier |
| age | Age of the individual |
| gender | Gender of the individual |
| height | Height in centimeters |
| weight | Weight in kilograms |
| ap_hi | Systolic blood pressure |
| ap_lo | Diastolic blood pressure |
| cholesterol | Cholesterol level |
| gluc | Glucose level |
| smoke | Smoking status |
| alco | Alcohol consumption status |
| active | Physical activity status |
| occupation | Occupation category |
| country | Country information |
| date | Record date |
| disease | Target variable indicating disease status |

**Target Variable:**

- `0` – No heart disease indicated.
- `1` – Heart disease indicated.

## Technologies & Tools Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| Scikit-learn | Machine Learning algorithms |
| Logistic Regression | Classification algorithm |
| Decision Tree | Classification algorithm |
| Random Forest | Ensemble learning algorithm |
| Support Vector Machine | Classification algorithm |
| StandardScaler | Numerical feature scaling |
| SimpleImputer | Handling missing values |
| OneHotEncoder | Categorical feature encoding |
| Pickle | Saving and loading trained models |
| Streamlit | Interactive web application |
| Jupyter Notebook | Development environment |
| Git & GitHub | Version control and project hosting |

## Project Workflow

### 1. Data Collection

- Imported the dataset using Pandas.
- Loaded the CSV file into a DataFrame.
- Examined the first few records using `head()`.
- Analyzed statistical information using `describe()`.
- Inspected data types and dataset information using `info()`.
- Checked dataset dimensions and column names.

### 2. Data Cleaning

Performed data cleaning to improve dataset quality.

The following operations were implemented:

- Converted age from days into years.
- Removed irrelevant columns such as `id` and `date`.
- Converted gender into numerical values.
- Checked for missing values.
- Removed records containing missing values in the application pipeline's dataset preparation.
- Filtered invalid blood pressure values.

**Blood Pressure Filtering:**

- Systolic blood pressure (`ap_hi`) greater than 60 and less than 250.
- Diastolic blood pressure (`ap_lo`) greater than 30 and less than 200.

These filters help remove potentially invalid blood pressure records.

### 3. Feature Engineering

Applied feature engineering to prepare the dataset for Machine Learning.

**Age Conversion:**

Converted age from days into years:

```python
df['age'] = (df['age'] / 365).astype(int)
```

**Categorical Encoding:**

Converted categorical information into numerical representations.

Gender encoding:

- Male → 1
- Female → 0

Applied one-hot encoding to categorical features such as occupation and country in the initial preprocessing workflow.

### 4. Feature Scaling

Implemented `StandardScaler` to standardize numerical features.

Numerical features included:

- Age
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Weight
- Height

Feature scaling helps numerical features contribute on comparable scales during model training.

### 5. Train-Test Split

Separated the dataset into training and testing datasets using Scikit-learn's `train_test_split()`.

| Parameter | Value |
|---|---|
| Training Data | 80% |
| Testing Data | 20% |
| Random State | 42 |
| Split Method | Train-Test Split |

The training dataset is used to train classification models, while the testing dataset is used to evaluate their performance.

### 6. Handling Missing Values

Implemented `SimpleImputer` with the mean strategy to handle missing numerical values.

The imputer was used in the Logistic Regression and SVM workflows.

For the final prediction pipeline, missing records were removed during dataset preparation.

### 7. Machine Learning Model Development

Implemented four classification algorithms to predict heart disease.

**Algorithms Used:**

**1. Logistic Regression**

A supervised classification algorithm used to predict binary outcomes based on input features.

**2. Decision Tree Classifier**

A tree-based algorithm that makes predictions by splitting data according to feature conditions.

**3. Random Forest Classifier**

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

**4. Support Vector Machine (SVM)**

A supervised learning algorithm that identifies an optimal decision boundary between different classes.

Each algorithm was trained and evaluated using the prepared dataset.

### 8. Model Evaluation

Evaluated the classification models using the following performance metrics:

- Accuracy Score
- Precision Score
- Recall Score
- F1-Score
- Confusion Matrix
- Classification Report

These evaluation metrics help understand how effectively the models classify disease and non-disease records.

**Evaluation Metrics Explained:**

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall correct predictions |
| Precision | Measures how many predicted positive cases are actually positive |
| Recall | Measures how many actual positive cases are correctly identified |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | Shows correct and incorrect classification results |

### 9. Model Saving

Used Python's Pickle library to save the trained Random Forest model.

Saved model:

`heart_disease_model.pkl`

Also developed a complete preprocessing and prediction pipeline.

Saved pipeline:

`heart_disease_pipeline.pkl`

The pipeline combines:

- ColumnTransformer
- StandardScaler
- OneHotEncoder
- RandomForestClassifier

This allows preprocessing and prediction to be performed together when new customer information is provided.

### 10. Streamlit Web Application

Developed an interactive web application using Streamlit.

The application is titled:

**Heart Disease Predictor**

Users can enter their health-related information through interactive input fields.

**Input Features:**

- Age
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Weight
- Height
- Cholesterol
- Glucose
- Gender
- Smoking Status
- Alcohol Consumption
- Physical Activity
- Occupation
- Country

**Application Workflow:**

1. Load the saved Machine Learning pipeline.
2. Collect health-related information from the user.
3. Convert the entered information into a Pandas DataFrame.
4. Convert gender into numerical representation.
5. Pass the input data into the trained pipeline.
6. Generate a prediction.
7. Display the prediction result.

**Prediction Output:**

- High Risk of Heart Disease
- Low Risk of Heart Disease

The application provides an interactive interface for demonstrating the trained Machine Learning model.

## Model Performance

The notebook implements four Machine Learning classification algorithms.

| Algorithm | Evaluation |
|---|---|
| Logistic Regression | Accuracy, Precision, Recall, F1-Score |
| Decision Tree | Accuracy, Precision, Recall, F1-Score |
| Random Forest | Accuracy, Precision, Recall, F1-Score |
| Support Vector Machine | Accuracy, Precision, Recall, F1-Score |


## Installation and Setup

### Prerequisites

- Python 3.x
- pip
- Jupyter Notebook

### Step 1: Clone the Repository

```bash
git clone https://github.com/SanthanamariG/Machine-Learning-Projects.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Early-Disease-Detection
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows:**

```powershell
venv\Scripts\activate
```

**Linux:**

```bash
source venv/bin/activate
```

### Step 5: Run Jupyter Notebook

```bash
jupyter notebook
```

Open `Early_Disease_Detection.ipynb` and execute the notebook cells.

### Step 6: Run Streamlit Application

```bash
streamlit run src/app.py
```

Open the local URL displayed in the terminal to access the application.

**Important:** Ensure that the saved model pipeline exists in the expected location before running the application.

## Future Improvements

- Improve prediction performance through hyperparameter tuning.
- Implement cross-validation.
- Perform feature importance analysis.
- Evaluate class imbalance and apply appropriate balancing techniques if required.
- Improve model generalization.
- Add additional evaluation visualizations.
- Improve the Streamlit application interface.
- Deploy the application on a cloud platform.
- Implement automated model training and deployment.

## Key Learnings

Through this project, I gained practical experience in:

- Working with health-related datasets.
- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- Feature engineering.
- Categorical encoding.
- Numerical feature scaling.
- Handling missing values.
- Implementing classification algorithms.
- Evaluating Machine Learning models.
- Building Scikit-learn pipelines.
- Saving and loading trained models.
- Developing interactive applications using Streamlit.
