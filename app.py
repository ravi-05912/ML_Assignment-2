
import streamlit as st
import pandas as pd
import joblib
import os

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from src.preprocessing import prepare_data


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")


# --------------------------------------------------
# Model File Mapping
# --------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}


# --------------------------------------------------
# Load Saved Models
# --------------------------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, file_name in MODEL_FILES.items():

        model_path = os.path.join(MODEL_DIR, file_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


# Load all five models
try:
    models = load_models()
    st.success("All five trained models loaded successfully.")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# --------------------------------------------------
# Application Title and Description
# --------------------------------------------------

st.title("Breast Cancer Classification")

st.write(
    """
    This application evaluates five machine learning
    classification models using the Breast Cancer
    Wisconsin Diagnostic dataset.

    Upload a test dataset to evaluate the models and
    compare their classification performance.
    """
)

st.info(
    "This application is intended for educational purposes "
    "only and must not be used for clinical diagnosis."
)


# --------------------------------------------------
# CSV File Upload
# --------------------------------------------------

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing test data",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        uploaded_data = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully.")

        st.subheader("Uploaded Dataset Preview")

        st.dataframe(uploaded_data.head())

        st.write("Dataset shape:", uploaded_data.shape)

    except Exception as e:

        st.error(f"Unable to read the uploaded CSV file: {e}")
        st.stop()


# --------------------------------------------------
# Validate and Prepare Uploaded Dataset
# --------------------------------------------------

if uploaded_file is not None:

    try:
        X_test, y_test = prepare_data(uploaded_data)

        # Check that the uploaded features match model features
        expected_features = list(models["Logistic Regression"].feature_names_in_)

        if list(X_test.columns) != expected_features:
            st.error(
                "The uploaded dataset does not contain "
                "the expected features in the correct order."
            )
            st.stop()

        st.success("Dataset validation completed successfully.")

        st.write("Number of input features:", X_test.shape[1])
        st.write("Number of test records:", X_test.shape[0])

    except Exception as e:

        st.error(f"Dataset validation failed: {e}")
        st.stop()



# --------------------------------------------------
# Model Selection
# --------------------------------------------------

st.header("Model Selection")

selected_model_name = st.selectbox(
    "Choose a classification model:",
    list(models.keys())
)

selected_model = models[selected_model_name]


# --------------------------------------------------
# Generate Predictions
# --------------------------------------------------

if uploaded_file is not None:

    # Predict class labels
    y_pred = selected_model.predict(X_test)

    # Obtain prediction scores for AUC
    if selected_model_name == "SVM":
        y_score = selected_model.decision_function(X_test)
    else:
        y_score = selected_model.predict_proba(X_test)[:, 1]

    st.success(
        f"Predictions generated using {selected_model_name}."
    )


# --------------------------------------------------
# Calculate Evaluation Metrics
# --------------------------------------------------

if uploaded_file is not None:

    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_score)

    precision = precision_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)


    # --------------------------------------------------
    # Display Evaluation Metrics
    # --------------------------------------------------

    st.header("Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

st.header("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(6, 4))

display_cm = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"]
)

display_cm.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

st.pyplot(fig)

plt.close(fig)


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

st.header("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df.round(4))


# --------------------------------------------------
# Compare All Five Models
# --------------------------------------------------

st.header("Comparison of All Five Models")

comparison_results = []

for model_name, model in models.items():

    # Generate predictions
    model_pred = model.predict(X_test)

    # Obtain scores for AUC
    if model_name == "SVM":
        model_score = model.decision_function(X_test)
    else:
        model_score = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    comparison_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, model_pred),
        "AUC": roc_auc_score(y_test, model_score),
        "Precision": precision_score(
            y_test, model_pred, pos_label=1, zero_division=0
        ),
        "Recall": recall_score(
            y_test, model_pred, pos_label=1, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, model_pred, pos_label=1, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, model_pred)
    })


# Convert comparison results into a DataFrame
comparison_df = pd.DataFrame(comparison_results)

# Display comparison table
st.dataframe(
    comparison_df.round(4),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# Identify Best Performing Model
# --------------------------------------------------

best_model = comparison_df.loc[
    comparison_df["F1 Score"].idxmax()
]

st.subheader("Best Performing Model (Based on F1 Score)")

st.success(
    f"{best_model['Model']} achieved the highest "
    f"F1 Score of {best_model['F1 Score']:.4f}."
)
