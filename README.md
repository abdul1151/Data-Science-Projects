Data Science Project: Comprehensive Analysis and Modeling
This repository contains four tasks that demonstrate data analysis, machine learning, and modeling techniques. Each task focuses on solving a specific real-world problem using Python.

Tasks Overview
Task 1: Exploratory Data Analysis (EDA) of Titanic Dataset
Perform exploratory data analysis to uncover insights about passenger survival on the Titanic.
Task 2: Sentiment Analysis on IMDB Dataset
Build a machine learning model to classify movie reviews as positive or negative.
Task 3: Fraud Detection System
Develop a fraud detection system to classify credit card transactions as fraudulent or non-fraudulent.
Task 4: Predicting House Prices Using the Boston Housing Dataset
Implement regression models to predict house prices based on various features.

Project Steps
Task 1: EDA on Titanic Dataset
Steps:
Load the Titanic dataset and clean missing values.
Visualize data using histograms, bar charts, and correlation heatmaps.
How to Run:
Execute the Jupyter script:
Run Cell by Cell
Observations:
Female passengers and first-class passengers had higher survival rates.
Higher ticket fares positively correlated with survival.
Task 2: Sentiment Analysis on IMDB Dataset
Steps:
Preprocess text data (tokenization, stopword removal, lemmatization).
Convert text to numerical data using TF-IDF.
Train Logistic Regression and Naive Bayes models.
Evaluate models using precision, recall, and F1-score.
How to Run:
Execute the Jupyter script:
Run Cell by Cell
Observations:
Logistic Regression outperformed Naive Bayes.
Positive reviews often used enthusiastic terms like "amazing," while negative reviews had strong criticism.
Task 3: Fraud Detection System
Steps:
Preprocess the dataset and address class imbalance using undersampling.
Train a Random Forest model to detect fraud.
Evaluate the model using precision, recall, and F1-score.
Build a simple interface to test individual transactions.
How to Run:
Execute the Jupyter script:
Run Cell by Cell
Observations:
The model effectively detected fraudulent transactions with a balanced performance.
Feature importance analysis revealed that certain transaction patterns strongly indicated fraud.
Task 4: Predicting House Prices Using the Boston Housing Dataset
Steps:
Load the Boston Housing dataset and normalize numerical features.
Implement Linear Regression, Random Forest, and XGBoost models from scratch.
Evaluate models using RMSE and R² metrics.
Visualize feature importance for tree-based models.
How to Run:
Execute the Jupyter script:
Run Cell by Cell
Observations:
Random Forest provided the best performance, followed by XGBoost.
Features like RM (average number of rooms) positively correlated with house prices.
LSTAT (lower-status population) negatively impacted house prices.
