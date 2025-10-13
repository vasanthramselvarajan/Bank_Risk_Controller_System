# Bank Loan Prediction and Risk Analysis

This repository contains a comprehensive project on predicting loan approvals and analyzing customer risk for a bank. The project involves data cleaning, exploratory data analysis (EDA), machine learning model training, and an interactive Streamlit app for presenting the results. Additionally, a movie recommendation system is integrated into the app.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Technologies Used](#technologies-used)

---

## Overview

The goal of this project is to predict loan approval and assess risk based on customer data. The project includes:
- Cleaning and preprocessing the dataset.
- Performing EDA to uncover insights.
- Training machine learning models to predict loan approval.
- Building an interactive Streamlit app to present the findings and predictions.
- Integrating a movie recommendation system for additional functionality.

---

## Project Structure

The repository contains the following files:

1. **`bank_datacleaning.ipynb`**:
   - Reads and preprocesses the raw dataset (`loan_data.csv`).
   - Handles missing values, reduces dimensionality, and performs feature selection using different statistical test.
   - Used different approches for continous and catagorical features.
   - Outputs a cleaned dataset (`df_cleaned.csv`) for modeling.

2. **`bank_eda_model.ipynb`**:
   - Performs detailed EDA, including visualizations for age distribution, income vs. credit, and target distribution ect..
   - Implements feature engineering, such as creating new features like `INCOME_TO_CREDIT` and age categories.
   - Trains machine learning models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting ect..).
   - Evaluates models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

3. **`bank_app.py`**:
   - A Streamlit app for interactive exploration and prediction.
   - Includes sections for:
     - **EDA Visualizations**: Interactive charts for data insights.
     - **Model Performance**: Displays metrics for trained models.
     - **Loan Prediction**: Predicts loan approval based on user input.
     - **Movie Recommendation**: Suggests movies based on user preferences.
   - Utilizes pre-trained models and label encoders stored as `.pkl` files.

4. **Movie Recommendation System**:
   - Integrated into the Streamlit app.
   - Provides personalized movie recommendations based on user preferences.

---

## Features

- **Data Cleaning**:
  - Handles missing values and reduces dimensionality.
  - Creates a well cleaned dataset for training.

- **Exploratory Data Analysis**:
  - Visualizes key insights using Matplotlib, Seaborn, and Plotly.
  - Analyzes relationships between features and the target variable.

- **Machine Learning**:
  - Trains multiple models to predict loan approval.
  - Evaluates models using metrics like accuracy, precision, recall, and F1-score.
  - Best performing model used in live prediction.

- **Interactive App**:
  - Displays EDA visualizations and model performance.
  - Allows users to input data for loan prediction.
  - Includes a movie recommendation feature.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bank-loan-prediction.git
   cd bank-loan-prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run bank_app.py
   ```

---

## Usage

1. **Data Cleaning**:
   - Open `bank_datacleaning.ipynb` to preprocess the dataset.

2. **EDA and Model Training**:
   - Use `bank_eda_model.ipynb` to explore the data and train models.

3. **Interactive App**:
   - Run `bank_app.py` to launch the Streamlit app to interact with the project.

4. **Movie Recommendation system**:
   - Use `movie_recommendation.ipynb` to create a content based movie recommendation system.

---

## Technologies Used

- **Languages**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, Streamlit
- **Machine Learning**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting ect..
- **Other Tools**: Jupyter Notebook, Pickle

---
Feel free to contribute to this project by submitting issues or pull requests. For any questions, contact [vasanthramselvarajan2212@gmail.com](mailto:your-email@example.com).
