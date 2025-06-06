import pandas as pd
import joblib
import os
import argparse
import yaml
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import json

def evaluate_model(model_path, train_path, test_path):
    model = joblib.load(model_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    X = full_df.drop(columns=["posttest"])
    y_true = full_df["posttest"]

    y_pred = model.predict(X)

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics_decision_tree_fullset.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate_model(
        model_path="models/decision_tree_model.pkl",
        train_path=config["data_split"]["trainset_path"],
        test_path=config["data_split"]["testset_path"]
    )
