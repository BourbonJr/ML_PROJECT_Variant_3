import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json

def train_model(train_path, output_path, model_params):
    df = pd.read_csv(train_path)
    X = df.drop(columns=["posttest"])
    y = df["posttest"]

    model = DecisionTreeRegressor(**model_params)
    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = {
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": mean_squared_error(y, y_pred) ** 0.5,
        "R2": r2_score(y, y_pred)
    }

    os.makedirs("metrics_train", exist_ok=True)
    with open("metrics_train/metrics_decision_tree_train.json", "w") as f:
        json.dump(metrics, f, indent=4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)

if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    params = config["models"]["decision_tree"]
    train_model(
        train_path=config["data_split"]["trainset_path"],
        output_path="models/decision_tree_model.pkl",
        model_params=params
    )
    
