from pathlib import Path
import sys

# Fix import path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.load_data import load_communities_crime
from data.preprocessing import preprocess_dataset

# 🔥 THIS IS THE IMPORTANT PART
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "communities.data"


def main():
    print("Looking for file at:", DATA_PATH)

    df = load_communities_crime(DATA_PATH)

    processed = preprocess_dataset(
        df,
        missing_threshold=0.30,
        random_state=42,
        add_bias=True,
    )

    print("Train shape:", processed["X_train"].shape)
    print("Validation shape:", processed["X_val"].shape)
    print("Test shape:", processed["X_test"].shape)
    print("Number of features:", len(processed["feature_names"]))
    print("Dropped columns:", processed["dropped_columns"])


if __name__ == "__main__":
    main()