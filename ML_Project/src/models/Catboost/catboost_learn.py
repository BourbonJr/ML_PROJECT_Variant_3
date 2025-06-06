import pandas as pd
from catboost import CatBoostRegressor
import joblib
import os

def train_model(train_path, output_path, model_params):
    df = pd.read_csv(train_path)
    X = df.drop(columns=["posttest"])
    y = df["posttest"]

    model = CatBoostRegressor(verbose=0, **model_params)
    model.fit(X, y)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)

if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    params = config["models"]["catboost"]
    train_model(
        train_path=config["data_split"]["trainset_path"],
        output_path="models/catboost_model.pkl",
        model_params=params
    )
