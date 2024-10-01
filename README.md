# Credit Risk Analysis Using Machine Learning Models
## Overview
This repository contains the code and methodology for Credit Risk Analysis, specifically aimed at predicting loan defaults. The project explores the use of various machine learning models in combination with different feature engineering techniques to enhance the accuracy of loan default predictions. By leveraging the "Home Credit Default Risk" dataset, this analysis evaluates models such as Logistic Regression, Random Forest, XGBoost, K-Nearest Neighbors, Decision Trees, and AdaBoost.

The project compares original features to those transformed using Weight of Evidence (WOE) to examine the impact of feature engineering on model performance. The results of this study offer practical insights for financial institutions like banks and credit lenders, helping them make more informed decisions and manage credit risk more effectively.

## Project Motivation
Financial institutions face a continuous challenge of balancing customer acquisition with minimizing loan default risks. Traditional credit risk models often struggle to capture the complexity of borrower behavior. The motivation behind this project is to improve upon conventional credit scoring methods by applying advanced machine learning algorithms, with the goal of enhancing the precision and accuracy of loan default predictions.

## Dataset
The dataset used in this project is from the Home Credit Default Risk competition on Kaggle, which includes over 300,000 loan applications with numerous features related to customer demographics, loan characteristics, and financial data.

The data is publicly available and can be found here.
A subset of 100,000 rows and 46 relevant features was selected for analysis. The target variable is TARGET, a binary indicator where 1 represents a default and 0 represents non-default.
## Methodology
### 1. Data Preprocessing
- Anomaly Handling: We detected and removed rows with anomalous values in fields such as DAY_EMPLOYED and CODE_GENDER.
- Handling Missing Values: 21 columns contained missing values, two of which exceeded a 50% missing rate and were removed. The remaining missing values were imputed using statistical methods (mean for numerical features, mode for categorical).
- Feature Scaling: The Min-Max Scaler was applied to normalize feature distributions within a range of [0,1].
- Imbalance Handling: The dataset was imbalanced with a 0.0888 ratio of default to non-default. We used the SMOTE (Synthetic Minority Over-sampling Technique) to create a balanced dataset.
### 2. Feature Engineering
- Weight of Evidence (WOE) and Information Value (IV): WOE was employed to transform categorical features into numeric values, providing a better distinction between defaulters and non-defaulters. IV was used to assess the predictive power of each feature.
- Encoding: One-Hot Encoding was applied to categorical variables to make them usable by machine learning models.
### 3. Machine Learning Models
The following models were evaluated using both original features and WOE-transformed features:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forest
- XGBoost
- AdaBoost
Each model was trained and tested on the dataset, split into an 80:20 ratio for training and testing.

### 4. Evaluation Metrics
To assess the performance of the models, we used the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

These metrics help evaluate both the overall accuracy and the ability of the model to minimize false positives and false negatives.

## Results

- XGBoost emerged as the best-performing model across most metrics, particularly in terms of ROC-AUC, accuracy, and F1-Score.
- The use of original features resulted in slightly better performance compared to WOE-transformed features. However, WOE features still showed strong predictive power, especially for Logistic Regression.
- SMOTE significantly improved the model’s performance on the imbalanced dataset by ensuring balanced class distribution.
## Limitations
While the models achieved high accuracy and good ROC-AUC scores, other metrics such as precision and recall were lower than expected. This could be due to:

- Suboptimal hyperparameters: Further tuning may improve the balance across various evaluation metrics.
- Sampling issues: Alternative sampling techniques could be explored to improve model generalization.
## Conclusion
This project demonstrates the power of machine learning in credit risk analysis, with XGBoost proving to be the most effective model in predicting loan defaults. While feature engineering using WOE provided some improvements, the original features performed slightly better across most models. The study highlights the importance of model selection, feature engineering, and data preprocessing in building effective credit scoring models.

**Future research could include:**

- Exploring deep learning models such as Neural Networks and Support Vector Machines.
- Testing the entire Home Credit dataset for more comprehensive insights.
- Applying alternative data sources (e.g., social media) for more robust credit risk predictions.
## Usage

1. Python 3
2. Scikit-learn
3. XGBoost
4. RandomForest
5. SMOTE
6. Pandas
7. Numpy
8. Matplotlib