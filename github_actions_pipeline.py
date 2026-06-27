import os
import json
import joblib
import numpy as np
import pandas as pd

from datasets import load_dataset
from huggingface_hub import login, HfApi
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def main():
    hf_token = require_env("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME", "suhas005")

    dataset_repo = f"{hf_username}/tourism-wellness-dataset"
    model_repo = f"{hf_username}/tourism-wellness-model"
    space_repo = f"{hf_username}/tourism-wellness-app"

    login(token=hf_token)
    api = HfApi()

    print("Loading prepared train/test splits from Hugging Face dataset...")
    train_df = load_dataset(dataset_repo, "prepared", split="train").to_pandas()
    test_df = load_dataset(dataset_repo, "prepared", split="test").to_pandas()

    if "ProdTaken" not in train_df.columns:
        raise ValueError("ProdTaken column not found in train split.")
    if "ProdTaken" not in test_df.columns:
        raise ValueError("ProdTaken column not found in test split.")

    y_train = train_df["ProdTaken"].astype(int)
    X_train = train_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"].astype(int)
    X_test = test_df.drop(columns=["ProdTaken"])

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 8, 15],
        "model__min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    print("Training model with GridSearchCV...")
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "best_cv_f1": float(grid.best_score_),
    }

    print("Metrics:", metrics)

    os.makedirs("tourism_project/model_building/artifacts", exist_ok=True)
    model_path = "tourism_project/model_building/artifacts/best_model.joblib"
    metrics_path = "tourism_project/model_building/artifacts/metrics.json"
    config_path = "tourism_project/model_building/artifacts/model_config.json"

    joblib.dump(best_model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(config_path, "w") as f:
        json.dump({"best_params": grid.best_params_}, f, indent=2)

    print("Uploading model artifacts to Hugging Face Model Hub...")
    api.create_repo(repo_id=model_repo, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=model_path, path_in_repo="best_model.joblib", repo_id=model_repo, repo_type="model")
    api.upload_file(path_or_fileobj=metrics_path, path_in_repo="metrics.json", repo_id=model_repo, repo_type="model")
    api.upload_file(path_or_fileobj=config_path, path_in_repo="model_config.json", repo_id=model_repo, repo_type="model")

    print("Creating deployment files...")
    os.makedirs("tourism_project/deployment", exist_ok=True)

    dockerfile = """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
"""
    app_py = f"""import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
import joblib

st.set_page_config(page_title="Tourism Wellness Predictor", layout="centered")
st.title("Wellness Tourism Package Purchase Prediction")

@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id="{model_repo}", filename="best_model.joblib", repo_type="model")
    return joblib.load(path)

model = load_model()
st.write("Enter sample input values and click Predict.")

age = st.number_input("Age", 18, 100, 35)
citytier = st.selectbox("CityTier", [1, 2, 3])
monthlyincome = st.number_input("MonthlyIncome", 1000, 1000000, 30000)
pitchsatisfactionscore = st.slider("PitchSatisfactionScore", 1, 5, 3)

input_df = pd.DataFrame([{{"Age": age, "CityTier": citytier, "MonthlyIncome": monthlyincome, "PitchSatisfactionScore": pitchsatisfactionscore}}])

if st.button("Predict"):
    # Add missing columns expected by the trained pipeline with default placeholders
    expected_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else input_df.columns
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"Prediction: {{'Will Purchase' if pred == 1 else 'Will Not Purchase'}}")
    st.info(f"Purchase Probability: {{prob:.2%}}")
"""
    reqs = """streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
huggingface_hub==0.24.6
"""

    with open("tourism_project/deployment/Dockerfile", "w") as f:
        f.write(dockerfile)
    with open("tourism_project/deployment/app.py", "w") as f:
        f.write(app_py)
    with open("tourism_project/deployment/requirements.txt", "w") as f:
        f.write(reqs)

    print("Uploading deployment folder to Hugging Face Space...")
    api.create_repo(repo_id=space_repo, repo_type="space", private=False, space_sdk="streamlit", exist_ok=True)
    api.upload_folder(folder_path="tourism_project/deployment", repo_id=space_repo, repo_type="space")

    print(f"Done. Model: https://huggingface.co/{model_repo}")
    print(f"Done. Space: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
