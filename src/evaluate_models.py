
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models using six classification metrics.
    """

    results = []

    for model_name, model in models.items():

        # Generate predicted class labels
        y_pred = model.predict(X_test)

        # Obtain scores for AUC calculation
        if model_name == "SVM":
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)

        auc = roc_auc_score(y_test, y_score)

        precision = precision_score(
            y_test, y_pred, pos_label=1
        )

        recall = recall_score(
            y_test, y_pred, pos_label=1
        )

        f1 = f1_score(
            y_test, y_pred, pos_label=1
        )

        mcc = matthews_corrcoef(y_test, y_pred)

        # Store results
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc
        })

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df
