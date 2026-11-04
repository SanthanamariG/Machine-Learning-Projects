Customer Churn Prediction Using Machine Learning
A machine learning project that predicts whether a telecom customer is likely to leave a service.
Features
- Cleaned and prepared 7,043 customer records (7,032 after removing 11 rows with missing TotalCharges).
- Applied categorical encoding, one-hot encoding, and feature scaling.
- Compared Logistic Regression, Random Forest, XGBoost, SVM, and KNN.
- Built a Streamlit interface that accepts 19 customer inputs and displays a churn prediction.
Results
Best recorded comparison: Logistic Regression
- Accuracy: 80.38%
- Precision: 64.85%
- Recall: 57.22%
- F1-score: 60.80%

- The notebook separately saves a Random Forest model for the Streamlit workflow; the Logistic Regression metrics above are not the measured performance of that saved app  model.
Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn, Joblib.
Run
Install the required packages, run the notebook to prepare the data and model files, then start the app:
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn joblib
streamlit run app.py
