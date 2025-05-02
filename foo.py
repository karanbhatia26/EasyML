from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# import autosklearn.classification
from tpot import TPOTClassifier
import numpy as np
import pandas as pd

data = fetch_openml(name='adult', version=2, as_frame=True)
X = data.data
y = data.target

y = y.astype(str)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical and numerical features
categorical_cols = X.select_dtypes(include=['category', 'object']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Define param grid first
param_grid = {
    "classifier": [LogisticRegression(max_iter=1000), RandomForestClassifier()],
    "classifier__C": [0.1, 1.0, 10.0]
}

# We cannot set conditional params easily, so we do GridSearch only for LogisticRegression for now
logreg_grid = {
    "classifier__C": [0.1, 1.0, 10.0]
}

# Run Grid Search
grid_search = GridSearchCV(Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
]), logreg_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_preds = grid_search.predict(X_test)
grid_acc = accuracy_score(y_test, grid_preds)

# Run Random Search
random_search = RandomizedSearchCV(Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
]), {"classifier__C": np.logspace(-2, 2, 10)}, n_iter=5, cv=3, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)
random_preds = random_search.predict(X_test)
random_acc = accuracy_score(y_test, random_preds)

# automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=600, per_run_time_limit=60)
# automl.fit(X_train, y_train)
# autosklearn_preds = automl.predict(X_test)
# autosklearn_acc = accuracy_score(y_test, autosklearn_preds)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, max_time_mins=10, random_state=42)
tpot.fit(X_train, y_train)
tpot_preds = tpot.predict(X_test)
tpot_acc = accuracy_score(y_test, tpot_preds)

grid_acc, random_acc, tpot_acc