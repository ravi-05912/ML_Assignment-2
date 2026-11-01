Breast Cancer Classification Using Machine Learning

1) Project Overview
This project implements and compares five supervised machine learning classification  algorithms to predict whether a breast tumour is Benign or Malignant using the Breast Cancer Wisconsin (Diagnostic) dataset.

The objective is to evaluate the performance of different classification algorithms using multiple evaluation metrics and deploy the trained models through an interactive Streamlit web application.

2) Problem Statement
Breast cancer diagnosis involves distinguishing between benign and malignant tumours based on characteristics extracted from digitized images of fine needle aspirates (FNA) of breast masses.

This project uses machine learning classification techniques to predict the diagnosis based on 30 numerical input features.

The target variable has two classes:
     - Benign (B): Non-malignant tumour.
     - Malignant (M): Malignant tumour.

Malignant is treated as the positive class during model evaluation.

3) Dataset Description

Dataset: Breast Cancer Wisconsin (Diagnostic)
Source: UCI Machine Learning Repository
Dataset URL: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Total instances         -   569
Input features          -   30
Target variable         -    diagnosis
Classification type     -	Binary classification
Missing values          -   0
Duplicate records       -	0

The dataset contains 30 numerical features describing characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity and symmetry.
The original target labels were mapped as follows:
    -   Benign (B) = 0
    -   Malignant (M) = 1

4) Data Preprocessing
The following preprocessing steps were performed:
    a. Separated the input features and target variable.
    b. Converted target labels into numerical values.
    c. Divided the dataset into training (80%) and testing (20%) sets.
    d. Used stratified sampling to preserve the original class distribution.
    e. Applied StandardScaler to Logistic Regression, KNN and SVM through machine learning pipelines.
The Decision Tree and Gaussian Naive Bayes models were trained without feature scaling.
The training set contains 455 records, while the test set contains 114 records.

5) Machine Learning Models
The following five classification algorithms were implemented:
1.	Logistic Regression
2.	Decision Tree Classifier
3.	K-Nearest Neighbors (KNN)
4.	Gaussian Naive Bayes
5.	Support Vector Machine (SVM) with RBF kernel

All five models were trained using the same training dataset and evaluated using the same held-out test dataset.

6) Evaluation Metrics

The models were evaluated using six performance metrics:
 - Accuracy: Overall proportion of correctly classified observations.
 - AUC: Measures the model's ability to distinguish between the two classes across classification thresholds.
 - Precision: Proportion of predicted malignant cases that were actually malignant.
 - Recall: Proportion of actual malignant cases correctly identified.
 - F1 Score: Harmonic mean of precision and recall.
 - MCC: Matthews Correlation Coefficient, which considers all four confusion matrix outcomes.



