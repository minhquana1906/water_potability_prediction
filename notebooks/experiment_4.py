import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
import pickle
from loguru import logger
from pathlib import Path
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
)
from src.config import DATA_DISK, CONFUSION_MATRIX_DIR, ROC_CURVE_DIR, MODELS_DIR

model_path: Path = MODELS_DIR / "model.pkl"
cm_path: Path = CONFUSION_MATRIX_DIR / "confusion_matrix.png"
roc_path: Path = ROC_CURVE_DIR / "roc_curve.png"

dagshub.init(repo_owner="minhquana1906", repo_name="water_potability_prediction", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/minhquana1906/water_potability_prediction.mlflow")
mlflow.set_experiment("Experiment 4")


def fill_missing_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.mean())


data = pd.read_csv(DATA_DISK)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

X_train = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns=["Potability"], axis=1)
y_test = test_processed_data["Potability"]

model = RandomForestClassifier()

# Implement RandomizedSearchCV for hyperparameter tuning
param_dist = {
    "n_estimators": [50, 200, 500, 800, 1000, 1200],
    "max_depth": [None, 3, 9, 15, 30],
    "min_samples_split": [2, 8, 15, 100],
    "min_samples_leaf": [1, 2, 5],
}

search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42,
    scoring="accuracy",
)


with mlflow.start_run(run_name=f"Random Forest Tuning") as parent_run:
    logger.info(f"Training Random Forest model with hyperparameter tuning...")
    search.fit(X_train, y_train)
    for i in range(len(search.cv_results_["params"])):
        with mlflow.start_run(run_name=f"Combination {i}", nested=True) as children_run:
            mlflow.log_params(search.cv_results_["params"][i])
            mlflow.log_metric("mean_test_score", search.cv_results_["mean_test_score"][i])

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info("Logging best params to MLflow...")
    mlflow.log_params(search.best_params_)

    logger.info(f"Training model with parameters: {search.best_params_}")
    # model.set_params(**search.best_params_)
    model = search.best_estimator_
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    logger.info(f"Evaluating model...")
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Logging metrics to MLflow...")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1 score", f1)

    # mlflow.sklearn.log_model(model, model_name)

    logger.info(f"Saving model to {model_path}...")
    pickle.dump(model, open(model_path, "wb"))

    logger.info(f"Logging model to MLflow...")
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "RandomForestClassifier", signature=signature)

    logger.info(f"Generating confusion matrix and ROC curve and logging to MLflow...")
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot(
        cmap="Blues",
        values_format="d",
    )
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    roc = roc_auc_score(y_test, y_pred)
    fpr, tpr = roc_curve(y_test, y_pred)[:2]
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.savefig(roc_path)
    mlflow.log_artifact(roc_path)

    logger.info(f"Logging source code file to MLflow...")
    mlflow.log_artifact(__file__)

    logger.info(f"Logging dataset to MLflow...")
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)
    mlflow.log_input(train_df, "train")
    mlflow.log_input(test_df, "test")

    logger.success(f"Experiment completed successfully!")
