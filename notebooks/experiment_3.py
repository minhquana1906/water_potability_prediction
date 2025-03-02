import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import dagshub
import pickle
from loguru import logger
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
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
from config import DATA_DISK, CONFUSION_MATRIX_DIR, ROC_CURVE_DIR, MODELS_DIR

dagshub.init(repo_owner="minhquana1906", repo_name="water_potability_prediction", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/minhquana1906/water_potability_prediction.mlflow")
mlflow.set_experiment("Experiment 3")


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

models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(),
}
with mlflow.start_run(run_name=f"Experiment Run {len(mlflow.search_runs()) + 1}"):
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            # model_path: Path = MODELS_DIR / f"{model_name.replace('_', ' ').lower()}.pkl"
            # cm_path: Path = (
            #     CONFUSION_MATRIX_DIR
            #     / f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png"
            # )
            # roc_path: Path = (
            #     ROC_CURVE_DIR / f"{model_name.replace(' ', '_').lower()}_roc_curve.png"
            # )

            logger.info(f"Training {model_name} model...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1 score", f1)

            # mlflow.sklearn.log_model(model, model_name)

            # pickle.dump(model, open(model_path, "wb"))

            # cm = confusion_matrix(y_test, y_pred)
            # cm_display = ConfusionMatrixDisplay(cm).plot(
            #     cmap="Blues",
            #     values_format="d",
            # )
            # plt.savefig(cm_path)
            # mlflow.log_artifact(cm_path)

            # roc = roc_auc_score(y_test, y_pred)
            # fpr, tpr = roc_curve(y_test, y_pred)[:2]
            # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
            # plt.savefig(roc_path)
            # mlflow.log_artifact(roc_path)

            logger.success(f"Experiment {len(mlflow.search_runs())} completed successfully!")
