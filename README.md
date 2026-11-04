# Breast Cancer Classification Using Machine Learning

## 1. Project Overview
This project implements and compares five supervised machine learning classification algorithms to predict whether a breast tumour is **Benign** or **Malignant** using the Breast Cancer Wisconsin (Diagnostic) dataset. The objective is to evaluate the performance of different classification algorithms using multiple evaluation metrics and deploy the trained models through an interactive Streamlit web application.

## 2. Problem Statement
Breast cancer diagnosis involves distinguishing between benign and malignant tumours based on characteristics extracted from digitized images of fine needle aspirates (FNA) of breast masses. 

This project uses machine learning classification techniques to predict the diagnosis based on 30 numerical input features. The target variable has two classes:
* **Benign (B):** Non-malignant tumour.
* **Malignant (M):** Malignant tumour.

*Note: Malignant is treated as the positive class during model evaluation.*

## 3. Dataset Description
* **Dataset:** Breast Cancer Wisconsin (Diagnostic)
* **Source:** UCI Machine Learning Repository
* **Dataset URL:** [UCI Repository Link](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

| Attribute | Description |
| --- | --- |
| **Total instances** | 569 |
| **Input features** | 30 |
| **Target variable** | diagnosis |
| **Classification type** | Binary classification |
| **Missing values** | 0 |
| **Duplicate records** | 0 |

The dataset contains 30 numerical features describing characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, and symmetry. 

The target labels are mapped as follows:
* **Benign (B)** = `0`
* **Malignant (M)** = `1`

## 4. Data Preprocessing
The following preprocessing workflow was applied:
1. Separated the input features and target variable.
2. Converted target labels into numerical values (B - 0, M - 1).
3. Divided the dataset into **training (80%)** and **testing (20%)** sets.
4. Used stratified sampling to preserve the original class distribution.
5. Applied StandardScaler to Logistic Regression, KNN, and SVM within machine learning pipelines.

*Note: Decision Tree and Gaussian Naive Bayes models were trained without feature scaling. The training set contains 455 records, while the test set contains 114 records.*

## 5. Machine Learning Models
Five classification algorithms were implemented and trained:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Gaussian Naive Bayes**
5. **Support Vector Machine (SVM)** with RBF kernel

All models were trained on the exact same training split and evaluated using the same held-out test dataset.

## 6. Evaluation Metrics
The models were evaluated using six standard metrics:
* **Accuracy:** Overall proportion of correctly classified observations.
* **AUC:** Measures the model's ability to distinguish between classes across classification thresholds.
* **Precision:** Proportion of predicted malignant cases that were actually malignant.
* **Recall:** Proportion of actual malignant cases correctly identified.
* **F1 Score:** Harmonic mean of precision and recall.
* **MCC:** Matthews Correlation Coefficient, considering all four confusion matrix outcomes.

## 7. Model Performance Comparison
Evaluation results obtained on the held-out test dataset:

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9649 | **0.9960** | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| **Decision Tree** | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| **KNN** | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| **Naive Bayes** | 0.9386 | 0.9934 | **1.0000** | 0.8333 | 0.9091 | 0.8715 |
| **SVM** | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

## 8. Key Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| **Logistic Regression** | • Achieved an accuracy of 96.49%.<br>• Obtained the highest AUC score of 0.9960.<br>• Demonstrated strong precision and recall.<br>• Provided excellent discrimination between benign and malignant cases. |
| **Decision Tree** | • Achieved an accuracy of 92.98%.<br>• Recorded the lowest overall performance among the five models in this experiment.<br>• Provides an interpretable classification approach. |
| **K-Nearest Neighbors** | • Achieved an accuracy of 95.61%.<br>• Demonstrated strong precision and AUC.<br>• Its performance depends on the choice of neighbours and feature scaling. |
| **Gaussian Naive Bayes** | • Achieved an accuracy of 93.86%.<br>• Obtained a precision of 1.0000, indicating no false positive predictions on the test set.<br>• Its recall of 0.8333 indicates that some malignant cases were incorrectly classified as benign. |
| **Support Vector Machine** | • Achieved the highest accuracy of 97.37%.<br>• Obtained the highest F1 score of 0.9630 and MCC of 0.9442.<br>• Achieved a precision of 1.0000 and recall of 0.9286.<br>• Demonstrated the strongest overall performance on this test split. |

## 9. Overall Conclusion
Among the five classifiers, the **Support Vector Machine (SVM)** achieved the best overall performance on the held-out test dataset across accuracy, F1 score, and MCC. **Logistic Regression** produced the highest AUC score. 

> **Note:** These results are derived from a single stratified train-test split. Performance may vary with different random seeds or cross-validation strategies.

## 10. Streamlit Web Application
An interactive Streamlit web application was developed to evaluate the trained models.

**Features:**
* Upload a custom test dataset in CSV format.
* Select any of the five pre-trained classification models.
* View metric scores (Accuracy, AUC, Precision, Recall, F1, MCC).
* Display the confusion matrix and full classification report.
* Compare model performances side-by-side.

*App Link:* [To be added after deployment]

## 11. Repository Structure
```text
Assignment_2_ML/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── dataset/
│   └── breast_cancer.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── notebooks/
│   └── assignment_2.ipynb
│
└── outputs/
    ├── model_comparison.csv
    └── confusion_matrices/
