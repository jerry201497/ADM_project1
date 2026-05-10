from pathlib import Path
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from data.load_data import load_communities_crime
from data.preprocessing import preprocess_dataset
from models.blr import BayesianLinearRegression


DATA_PATH = BASE_DIR / "data" / "raw" / "communities.data"


def main():
    df = load_communities_crime(DATA_PATH)
    data = preprocess_dataset(df)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    feature_names = data["feature_names"]

    model = BayesianLinearRegression(alpha=1.0, beta=1.0)
    model.fit_evidence(X_train, y_train)

    print(f"Optimized alpha: {model.alpha:.6f}")
    print(f"Optimized beta: {model.beta:.6f}")

    y_mean, y_std = model.predict(X_test, return_std=True)

    lower = y_mean - 1.96 * y_std
    upper = y_mean + 1.96 * y_std

    print("\nFirst 5 predictive intervals:")
    for i in range(5):
        print(f"Prediction: {y_mean[i]:.4f}, 95% CI: [{lower[i]:.4f}, {upper[i]:.4f}]")

    weights = list(zip(feature_names, model.m_N))
    weights = [item for item in weights if item[0] != "intercept"]
    weights.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 5 predictors:")
    for name, weight in weights[:5]:
        print(f"{name}: {weight:.4f}")


if __name__ == "__main__":
    main()