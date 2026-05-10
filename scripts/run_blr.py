from pathlib import Path
import sys

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
    model.fit_evidence(X_train, y_train, verbose=False)

    print(f"Optimized alpha: {model.alpha:.6f}")
    print(f"Optimized beta: {model.beta:.6f}")
    print(f"Log evidence: {model.log_evidence_:.6f}")
    print(f"Iterations: {model.n_iter_}")

    y_mean, y_std, lower, upper = model.predict(
        X_test,
        return_std=True,
        return_interval=True,
    )

    print("\nFirst 5 predictive intervals:")
    for i in range(5):
        print(
            f"Prediction: {y_mean[i]:.4f}, "
            f"Std: {y_std[i]:.4f}, "
            f"95% CI: [{lower[i]:.4f}, {upper[i]:.4f}]"
        )

    print("\nTop 5 predictors:")
    summary = model.coefficient_summary(
        feature_names=feature_names,
        top_k=5,
        exclude_intercept=True,
    )

    for row in summary:
        print(
            f"{row['feature']}: "
            f"mean={row['mean']:.4f}, "
            f"std={row['std']:.4f}, "
            f"95% CI=[{row['lower_95']:.4f}, {row['upper_95']:.4f}]"
        )


if __name__ == "__main__":
    main()