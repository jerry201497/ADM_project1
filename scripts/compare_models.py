from pathlib import Path
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from data.load_data import load_communities_crime
from data.preprocessing import preprocess_dataset
from models.blr import BayesianLinearRegression
from models.baselines import OrdinaryLeastSquares, RidgeRegression
from utils.metrics import rmse, mae, r2_score


DATA_PATH = BASE_DIR / "data" / "raw" / "communities.data"


def evaluate_model(name, y_true, y_pred):
    return {
        "model": name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    df = load_communities_crime(DATA_PATH)
    data = preprocess_dataset(
        df,
        missing_threshold=0.30,
        random_state=42,
        add_bias=True,
    )

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    results = []

    # OLS
    ols = OrdinaryLeastSquares().fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)
    results.append(evaluate_model("OLS", y_test, y_pred_ols))

    # Ridge: tune lambda on validation set
    lambdas = np.logspace(-4, 4, 30)
    best_lambda = None
    best_val_rmse = float("inf")

    for lam in lambdas:
        ridge = RidgeRegression(lam=lam).fit(X_train, y_train)
        y_val_pred = ridge.predict(X_val)
        val_rmse = rmse(y_val, y_val_pred)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_lambda = lam

    ridge = RidgeRegression(lam=best_lambda).fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    results.append(evaluate_model(f"Ridge lambda={best_lambda:.4f}", y_test, y_pred_ridge))

    # BLR
    blr = BayesianLinearRegression(alpha=1.0, beta=1.0)
    blr.fit_evidence(X_train, y_train)
    y_pred_blr, y_std_blr = blr.predict(X_test, return_std=True)
    results.append(evaluate_model("BLR evidence", y_test, y_pred_blr))

    # 95% predictive interval coverage
    lower = y_pred_blr - 1.96 * y_std_blr
    upper = y_pred_blr + 1.96 * y_std_blr
    coverage = np.mean((y_test >= lower) & (y_test <= upper))

    print("\nModel comparison:")
    for row in results:
        print(
            f"{row['model']:20s} "
            f"RMSE={row['RMSE']:.4f} "
            f"MAE={row['MAE']:.4f} "
            f"R2={row['R2']:.4f}"
        )

    print("\nBLR hyperparameters:")
    print(f"alpha = {blr.alpha:.6f}")
    print(f"beta  = {blr.beta:.6f}")

    print("\nBLR 95% predictive interval coverage:")
    print(f"coverage = {coverage:.4f}")

    print("\nBest ridge lambda:")
    print(f"lambda = {best_lambda:.6f}")


if __name__ == "__main__":
    main()