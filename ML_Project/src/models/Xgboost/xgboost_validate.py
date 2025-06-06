import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import json
import os

def evaluate_model(test_path, model_path):
    df = pd.read_csv(test_path)
    X_test = df.drop(columns=["posttest"])
    y_true = df["posttest"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics_xgboost.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate_model(
        test_path=config["data_split"]["testset_path"],
        model_path="models/xgboost_model.pkl"
    )