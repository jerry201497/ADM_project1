# ADM Project - Bayesian Linear Regression

This project implements a full Bayesian Linear Regression (BLR) pipeline from scratch using the Communities and Crime dataset from the UCI Machine Learning Repository.

The project includes:

- Data preprocessing
- Bayesian Linear Regression
- Evidence maximization
- Posterior predictive intervals
- Comparison with OLS and Ridge Regression
- Prior sensitivity analysis

## Dataset

The dataset used is:

- `communities.data`
- `communities.names`

Both files should be placed inside:

data/raw/

## Project Structure

src/
  data/
  models/
  utils/

scripts/

## Main Scripts
Run BLR with evidence maximization:
python -m scripts.run_blr
Compare OLS, Ridge and BLR:
python -m scripts.compare_models
Prior sensitivity analysis:
python -m scripts.prior_sensitivity

## Main libraries used:

numpy
pandas
scikit-learn
matplotlib

