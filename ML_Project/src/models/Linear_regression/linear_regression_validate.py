import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os

def evaluate_model(test_path, model_path):
    df = pd.read_csv(test_path)

    
    df = df.dropna()
    df_encoded = pd.get_dummies(df, drop_first=True)

    X_test = df_encoded.drop(columns=["posttest"])
    y_true = df_encoded["posttest"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred)
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics_linear_regression.json", "w") as f:
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
        model_path="models/linear_regression_model.pkl"
    )
