import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from data.load_data import load_communities_crime
from data.preprocessing import preprocess_dataset

DATA_PATH = BASE_DIR / "data" / "raw" / "communities.data"

def maximize_evidence(X, y, max_iter=100, tol=1e-5):
    """Finds the optimal alpha and beta using MacKay updates."""
    N, D = X.shape
    alpha, beta = 1.0, 1.0
    
    # Precompute eigenvalues for speed
    eigenvalues = np.linalg.eigvalsh(X.T @ X)
    
    for i in range(max_iter):
        alpha_old, beta_old = alpha, beta
        
        # 1. Update Covariance and Mean
        S_N_inv = alpha * np.eye(D) + beta * (X.T @ X)
        S_N = np.linalg.inv(S_N_inv)
        m_N = beta * S_N @ X.T @ y
        
        # 2. Update Gamma (effective number of parameters)
        gamma = np.sum((beta * eigenvalues) / (alpha + beta * eigenvalues))
        
        # 3. Update Alpha and Beta
        alpha = gamma / (m_N.T @ m_N)
        sse = np.sum((y - X @ m_N)**2)
        beta = (N - gamma) / sse
        
        # Check if they stopped changing
        if np.abs(alpha - alpha_old) < tol and np.abs(beta - beta_old) < tol:
            print(f"Convergence reached at iteration {i}")
            break
            
    return alpha, beta, m_N

def main():
    df = load_communities_crime(DATA_PATH)
    data = preprocess_dataset(df)
    
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    print("Starting Evidence Maximization to find optimal Alpha/Beta...")
    best_alpha, best_beta, best_m_N = maximize_evidence(X_train, y_train)
    
    print(f"Optimal Alpha (Prior Precision): {best_alpha:.4f}")
    print(f"Optimal Beta (Noise Precision): {best_beta:.4f}")
    
    # Check the "Top Predictors" with the tuned weights
    feature_names = data["feature_names"]
    weights = list(zip(feature_names, best_m_N))
    weights.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop 5 Predictors (Optimized):")
    for name, w in weights[:5]:
        print(f"{name}: {w:.4f}")

if __name__ == "__main__":
    main()