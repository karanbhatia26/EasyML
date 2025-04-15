from sklearn.preprocessing import OneHotEncoder
import logging
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, fetch_covtype
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def load_and_split_dataset(dataset_name):
    logging.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "adult":
        data = fetch_openml(name='adult', version=2, as_frame=True)
    elif dataset_name == "iris":
        data = fetch_openml(name='iris', version=1, as_frame=True)
    elif dataset_name == "digits":
        data = fetch_openml(name='mnist_784', version=1, as_frame=True)
    elif dataset_name == "covtype":
        data = fetch_covtype(as_frame=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X = data.data.copy()
    y = data.target.copy()

    # Encode target labels if categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    # Encode categorical features if present
    if X.select_dtypes(include=['object', 'category']).shape[1] > 0:
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # updated
        X_cat = pd.DataFrame(enc.fit_transform(X[cat_cols]))
        X_cat.columns = enc.get_feature_names_out(cat_cols)
        X_cat.index = X.index
        X = pd.concat([X.drop(columns=cat_cols), X_cat], axis=1)

    # 80-20 split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # From trainval, get 75% for train and 25% for val => final 60/20/20
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
def evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    models = {
        "Gradient Boosting": GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    for name, model in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        acc_train = accuracy_score(y_train, pipe.predict(X_train))
        acc_val = accuracy_score(y_val, pipe.predict(X_val))
        acc_test = accuracy_score(y_test, pipe.predict(X_test))

        logging.info(f"{name} - Train: {acc_train:.2f} | Val: {acc_val:.2f} | Test: {acc_test:.2f}")


if __name__ == "__main__":
    for dataset in ["iris", "adult", "digits", "covtype"]:
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_dataset(dataset)
            evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)
        except Exception as e:
            logging.error(f"Error loading dataset {dataset}: {e}")
