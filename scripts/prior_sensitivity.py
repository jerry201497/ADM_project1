from pathlib import Path
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from data.load_data import load_communities_crime
from data.preprocessing import preprocess_dataset
from models.blr import BayesianLinearRegression
from utils.metrics import rmse, mae, r2_score


DATA_PATH = BASE_DIR / "data" / "raw" / "communities.data"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def interval_coverage(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))


def main():
    df = load_communities_crime(DATA_PATH)
    data = preprocess_dataset(
        df,
        missing_threshold=0.30,
        random_state=42,
        add_bias=True,
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    # First fit evidence-optimized BLR to estimate beta.
    evidence_model = BayesianLinearRegression(alpha=1.0, beta=1.0)
    evidence_model.fit_evidence(X_train, y_train)

    fixed_beta = evidence_model.beta

    print("Evidence-optimized reference:")
    print(f"alpha = {evidence_model.alpha:.6f}")
    print(f"beta  = {evidence_model.beta:.6f}")
    print(f"log evidence = {evidence_model.log_evidence_:.6f}")

    # Prior sensitivity: weak prior -> strong prior
    alpha_values = np.logspace(-4, 6, 25)

    results = []

    for alpha in alpha_values:
        model = BayesianLinearRegression(alpha=alpha, beta=fixed_beta)
        model.fit(X_train, y_train)

        y_pred, y_std, lower, upper = model.predict(
            X_test,
            return_std=True,
            return_interval=True,
        )

        coef_summary = model.coefficient_summary(
            feature_names=feature_names,
            top_k=5,
            exclude_intercept=True,
        )

        top_features = [row["feature"] for row in coef_summary]
        coefficient_norm = np.linalg.norm(model.posterior_mean_[1:])

        row = {
            "alpha": alpha,
            "beta": fixed_beta,
            "log_evidence": model.log_evidence_,
            "rmse": rmse(y_test, y_pred),
            "mae": mae(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "coverage_95": interval_coverage(y_test, lower, upper),
            "coefficient_norm": coefficient_norm,
            "top_5_features": "; ".join(top_features),
        }

        results.append(row)

    output_path = REPORTS_DIR / "prior_sensitivity_results.csv"

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alpha",
                "beta",
                "log_evidence",
                "rmse",
                "mae",
                "r2",
                "coverage_95",
                "coefficient_norm",
                "top_5_features",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nPrior sensitivity results saved to:")
    print(output_path)

    print("\nSelected alpha values:")
    selected_indices = [0, 5, 10, 15, 20, 24]

    for idx in selected_indices:
        row = results[idx]
        print(
            f"alpha={row['alpha']:.4e} "
            f"RMSE={row['rmse']:.4f} "
            f"R2={row['r2']:.4f} "
            f"coverage={row['coverage_95']:.4f} "
            f"coef_norm={row['coefficient_norm']:.4f}"
        )
        print(f"Top features: {row['top_5_features']}")
        print()

    print("Evidence-optimal alpha:")
    print(f"{evidence_model.alpha:.6f}")
    
        # Optional plots for the report
    alphas = np.array([row["alpha"] for row in results])
    rmses = np.array([row["rmse"] for row in results])
    r2s = np.array([row["r2"] for row in results])
    coverages = np.array([row["coverage_95"] for row in results])
    coef_norms = np.array([row["coefficient_norm"] for row in results])
    log_evidences = np.array([row["log_evidence"] for row in results])

    plt.figure()
    plt.semilogx(alphas, rmses, marker="o")
    plt.xlabel("Prior precision alpha")
    plt.ylabel("Test RMSE")
    plt.title("Prior sensitivity: RMSE")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "prior_sensitivity_rmse.png", dpi=300)
    plt.close()

    plt.figure()
    plt.semilogx(alphas, r2s, marker="o")
    plt.xlabel("Prior precision alpha")
    plt.ylabel("Test R²")
    plt.title("Prior sensitivity: R²")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "prior_sensitivity_r2.png", dpi=300)
    plt.close()

    plt.figure()
    plt.semilogx(alphas, coef_norms, marker="o")
    plt.xlabel("Prior precision alpha")
    plt.ylabel("Coefficient norm")
    plt.title("Prior sensitivity: shrinkage effect")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "prior_sensitivity_coef_norm.png", dpi=300)
    plt.close()

    plt.figure()
    plt.semilogx(alphas, coverages, marker="o")
    plt.axhline(0.95, linestyle="--")
    plt.xlabel("Prior precision alpha")
    plt.ylabel("95% interval coverage")
    plt.title("Prior sensitivity: predictive coverage")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "prior_sensitivity_coverage.png", dpi=300)
    plt.close()

    plt.figure()
    plt.semilogx(alphas, log_evidences, marker="o")
    plt.axvline(evidence_model.alpha, linestyle="--")
    plt.xlabel("Prior precision alpha")
    plt.ylabel("Log marginal likelihood")
    plt.title("Evidence as a function of alpha")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "prior_sensitivity_log_evidence.png", dpi=300)
    plt.close()

    print("\nPlots saved to reports/")


if __name__ == "__main__":
    main()