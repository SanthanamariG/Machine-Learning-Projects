Early Disease Detection Using Machine Learning
A machine learning project that classifies cardiovascular disease risk from health and lifestyle information.
Features
- Analyzed a dataset containing 70,000 records.
- Prepared data through age conversion, categorical encoding, missing-value handling, feature filtering, and scaling.
- Evaluated Logistic Regression, Decision Tree, and Random Forest classifiers.
- Built a Streamlit application with 13 health-related input fields for risk classification.
Results
Recorded Logistic Regression test results:
- Accuracy: 72.37%
- Precision: 74.80%
- Recall: 67.35%
- F1-score: 70.88%
- Test records: 13,802
The Streamlit app uses a separately fitted Random Forest pipeline; these Logistic Regression metrics are not the measured performance of that app pipeline.

- Tech Stack
Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn.
