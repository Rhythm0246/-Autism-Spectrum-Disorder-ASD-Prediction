# Autism Spectrum Disorder (ASD) Prediction

This notebook documents the process of building a machine learning model to predict Autism Spectrum Disorder (ASD).

## 1. Importing the dependencies

This section imports all the necessary libraries for data manipulation, visualization, and model building.

## 2. Data Loading & Understanding

The data is loaded from a CSV file and an initial inspection is performed to understand its structure, shape, and data types.

- The `ID` and `age_desc` columns were dropped as they were not considered important for the prediction.
- Country names were fixed for consistency.
- Class imbalance in the target variable (`Class/ASD`) was identified.

## 3. Exploratory Data Analysis (EDA)

Univariate and bivariate analysis were performed to gain insights into the data.

- Distribution plots and box plots were generated for numerical columns (`age`, `result`) to understand their distribution and identify outliers.
- Count plots were generated for categorical columns to visualize their distribution.
- Missing values in the `ethnicity` and `relation` columns were handled by replacing them with 'Others'.
- Label encoding was applied to the categorical columns and the encoders were saved.
- A correlation heatmap was generated to visualize the relationships between features.

## 4. Data preprocessing

- Outliers in the `age` and `result` columns were handled by replacing them with the median.
- The data was split into training and testing sets.
- SMOTE (Synthetic Minority Oversampling Technique) was applied to the training data to address the class imbalance in the target variable.

## 5. Model Training

Three tree-based classifiers were trained with default parameters using 5-fold cross-validation on the SMOTE-resampled training data:

- Decision Tree
- Random Forest
- XGBoost

## 6. Model Selection & Hyperparameter Tuning

- Hyperparameter tuning was performed for each of the three models using `RandomizedSearchCV` with 5-fold cross-validation.
- The best performing model based on cross-validation accuracy was selected.
- The best model was saved as a pickle file.

## 7. Evaluation

The best model was evaluated on the test data using the following metrics:

- Accuracy score
- Confusion Matrix
- Classification Report
