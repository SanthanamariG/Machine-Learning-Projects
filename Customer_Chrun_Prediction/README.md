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

- Tech Stack: Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn, and Joblib.

- Execution: Install dependencies using pip install -r requirements.txt and launch the application   using streamlit run app.py.
