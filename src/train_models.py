
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def create_models():
    """
    Create and return all five classification models.
    """

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=10000,
                random_state=42
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5
            ))
        ]),

        "Naive Bayes": GaussianNB(),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                random_state=42
            ))
        ])
    }

    return models


def train_models(X_train, y_train):
    """
    Train all five models using the training dataset.
    """

    models = create_models()

    for model_name, model in models.items():

        print(f"Training {model_name}...")

        model.fit(X_train, y_train)

        print(f"{model_name} training completed.\n")

    return models
