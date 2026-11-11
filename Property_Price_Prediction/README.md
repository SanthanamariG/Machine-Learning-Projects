# Property Price Prediction Using Machine Learning

## Project Overview

Property Price Prediction is a Machine Learning project developed to predict house prices based on various housing characteristics, geographical information, and demographic features.

The project uses the California Housing Dataset to analyze factors that influence property values, including median income, housing age, total rooms, total bedrooms, population, households, geographical coordinates, and ocean proximity.

This project implements Simple Linear Regression and Multiple Linear Regression algorithms to understand the relationship between housing features and property prices.

Additionally, a Machine Learning preprocessing pipeline is developed using Scikit-learn, and an interactive web application is created using Streamlit to generate property price predictions based on user inputs.

## Problem Statement

Predicting property prices accurately is important for buyers, sellers, real estate companies, and property investors.

Property values depend on several factors, including geographical location, household income, housing characteristics, and surrounding infrastructure.

The objective of this project is to develop a Machine Learning regression model that analyzes housing-related information and predicts the median value of a property.

## Project Objectives

- Analyze housing datasets to identify factors influencing property prices.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Understand relationships between housing features and property values.
- Implement Simple Linear Regression.
- Implement Multiple Linear Regression.
- Apply feature scaling to numerical variables.
- Perform one-hot encoding for categorical variables.
- Build a Machine Learning preprocessing pipeline.
- Evaluate model performance using regression metrics.
- Visualize actual and predicted property prices.
- Save the trained Machine Learning model.
- Develop an interactive prediction application using Streamlit.

## Dataset Information

This project uses the California Housing Dataset.

The dataset contains information about housing characteristics, geographical locations, household demographics, and median house values in California.

**Dataset Details:**

| Attribute | Description |
|---|---|
| Dataset Name | California Housing Dataset |
| Data Format | CSV |
| Target Variable | median_house_value |
| Problem Type | Regression |
| Prediction Type | Continuous Numerical Value |
| Missing Values | Present in total_bedrooms |

### Dataset Features

| Feature | Description |
|---|---|
| longitude | Geographic longitude of the housing block |
| latitude | Geographic latitude of the housing block |
| housing_median_age | Median age of houses in the block |
| total_rooms | Total number of rooms |
| total_bedrooms | Total number of bedrooms |
| population | Total population in the housing block |
| households | Total number of households |
| median_income | Median household income |
| ocean_proximity | Proximity of the housing block to the ocean |
| median_house_value | Median house value (target variable) |

**Target Variable:**

`median_house_value`

The model predicts the median house value based on the selected housing characteristics.

## Technologies & Tools Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| Scikit-learn | Machine Learning algorithms |
| Linear Regression | Regression algorithm |
| StandardScaler | Numerical feature scaling |
| OneHotEncoder | Categorical feature encoding |
| ColumnTransformer | Combining preprocessing operations |
| Pipeline | Building an end-to-end ML workflow |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web application |
| Jupyter Notebook | Development environment |
| Git & GitHub | Version control and project hosting |

## Project Workflow

### 1. Data Collection

- Imported the California Housing Dataset using Pandas.
- Loaded the CSV file into a DataFrame.
- Examined the first few records using `head()`.
- Analyzed statistical information using `describe()`.
- Inspected dataset structure using `info()`.
- Checked missing values using `isnull().sum()`.

### 2. Data Cleaning

Performed initial data inspection and preprocessing.

The following operations were considered:

- Identified missing values in the `total_bedrooms` column.
- Calculated the median value of `total_bedrooms`.
- Examined numerical and categorical features.
- Prepared the dataset for Machine Learning model development.

**Missing Value Handling:**

The notebook calculates the median of `total_bedrooms` and calls `fillna()`.

For the imputation to take effect, the result should be assigned back to the column or performed using `inplace=True`.

### 3. Exploratory Data Analysis (EDA)

Performed Exploratory Data Analysis to understand the relationship between housing characteristics and property values.

**Visualizations Implemented:**

**1. Distribution of Median House Value**

Used a histogram with a KDE curve to understand the distribution of property prices.

**2. Median Income vs House Value**

Used a scatter plot to analyze the relationship between household income and median house value.

**3. Housing Median Age vs House Value**

Used a scatter plot to examine how housing age relates to property values.

**4. Correlation Heatmap**

Used a correlation heatmap to identify relationships between numerical features.

**5. Ocean Proximity vs House Value**

Used a box plot to understand how geographical proximity to the ocean influences median house values.

### 4. Feature Selection

Separated the independent variables and dependent variable.

**Independent Variables (Features):**

- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- ocean_proximity

**Dependent Variable (Target):**

- median_house_value

The target variable is separated from the input features before model training.

### 5. Train-Test Split

Divided the dataset into training and testing datasets using Scikit-learn's `train_test_split()`.

| Parameter | Value |
|---|---|
| Training Data | 80% |
| Testing Data | 20% |
| Random State | 42 |
| Splitting Method | Train-Test Split |

The training dataset is used to train the Machine Learning models, while the testing dataset is used to evaluate their prediction performance.

### 6. Simple Linear Regression

Implemented Simple Linear Regression using `LinearRegression()`.

In this approach, only one independent variable is used to predict the target variable.

**Selected Feature:**

`median_income`

**Workflow:**

1. Selected median income as the independent variable.
2. Trained the Linear Regression model.
3. Generated predictions using the testing dataset.
4. Calculated Mean Squared Error.
5. Calculated Root Mean Squared Error.
6. Evaluated model performance using R² Score.

This model helps understand the relationship between median household income and median house value.

### 7. Multiple Linear Regression

Implemented Multiple Linear Regression using several independent variables.

Unlike Simple Linear Regression, this model considers multiple housing characteristics simultaneously.

**Numerical Features:**

- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income

**Categorical Feature:**

- ocean_proximity

### 8. Data Preprocessing Pipeline

Developed a preprocessing pipeline using Scikit-learn.

**Numerical Feature Processing:**

Applied `StandardScaler()` to standardize numerical features.

**Categorical Feature Processing:**

Applied `OneHotEncoder(handle_unknown='ignore')` to convert categorical variables into numerical representations.

**ColumnTransformer:**

Combined numerical and categorical preprocessing operations using `ColumnTransformer`.

**Machine Learning Pipeline:**

Created an end-to-end pipeline containing:

1. Data preprocessing.
2. Numerical feature scaling.
3. Categorical feature encoding.
4. Multiple Linear Regression model.

This approach makes the preprocessing and prediction workflow easier to manage.

### 9. Model Evaluation

Evaluated both regression models using the following performance metrics:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

**Evaluation Metrics Explained:**

| Metric | Purpose |
|---|---|
| MSE | Measures the average squared difference between actual and predicted values |
| RMSE | Measures prediction errors in the same unit as the target variable |
| R² Score | Measures how much variation in the target variable is explained by the model |

### 10. Model Comparison

Compared the performance of Simple Linear Regression and Multiple Linear Regression.

| Model | Evaluation Metrics |
|---|---|
| Simple Linear Regression | MSE, RMSE, R² Score |
| Multiple Linear Regression | MSE, RMSE, R² Score |

The comparison helps identify how using multiple housing characteristics affects prediction performance compared with using only median income.

### 11. Prediction Visualization

Visualized actual and predicted median house values using a scatter plot.

The visualization compares:

- Actual Median House Values.
- Predicted Median House Values.

A reference line is included to help understand the difference between actual and predicted values.

### 12. Residual Analysis

Performed residual analysis to understand prediction errors.

**Residual Calculation:**

Residual = Actual Value - Predicted Value

Implemented the following visualizations:

- Residual Distribution Histogram.
- Residuals vs Predicted Values Scatter Plot.

These visualizations help identify prediction errors and assess the regression model's behavior.

### 13. Model Saving and Loading

Used Joblib to save and load the trained Multiple Linear Regression pipeline.

**Saved Model:**

`final_house_price_model.pkl`

**Saving the Model:**

```python
import joblib

joblib.dump(pipeline, 'final_house_price_model.pkl')
```

**Loading the Model:**

```python
loaded_model = joblib.load('final_house_price_model.pkl')
```

The saved pipeline includes preprocessing operations and the trained regression model.

This allows new property information to be processed and used for predictions without manually repeating the preprocessing steps.

## Streamlit Web Application

Developed an interactive web application using Streamlit.

The application is titled:

**California House Price Prediction**

Users can enter housing-related information through interactive input fields and generate property price predictions.

### Input Features

The application accepts the following inputs:

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

### Application Workflow

1. Load the saved Machine Learning pipeline.
2. Collect housing information from the user.
3. Convert user inputs into a Pandas DataFrame.
4. Pass the input data into the trained pipeline.
5. Generate the predicted house value.
6. Display the predicted property price.

### Prediction Output

The application displays the predicted median house value in US dollars.

Example output format:

`Predicted House Value: $250,000.00`

The actual prediction depends on the values entered by the user.

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
cd Property-Price-Prediction
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

Open `Property_Price_Prediction.ipynb` and execute the notebook cells.

### Step 6: Run Streamlit Application

```bash
streamlit run src/app.py
```

## Future Improvements

- Implement advanced regression algorithms such as Random Forest Regressor and XGBoost Regressor.
- Perform hyperparameter tuning.
- Implement cross-validation.
- Improve model prediction performance.
- Perform feature importance analysis.
- Handle missing values using a dedicated preprocessing pipeline.
- Add additional property-related features.
- Improve the Streamlit user interface.
- Deploy the application using a cloud platform.
- Implement automated Machine Learning model training and deployment.

## Key Learnings

Through this project, I gained practical experience in:

- Working with real-world housing datasets.
- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- Feature selection.
- Simple Linear Regression.
- Multiple Linear Regression.
- Numerical feature scaling.
- Categorical feature encoding.
- Building Scikit-learn preprocessing pipelines.
- Evaluating regression models.
- Visualizing actual and predicted values.
- Residual analysis.
- Saving and loading trained models.
- Developing interactive applications using Streamlit.
