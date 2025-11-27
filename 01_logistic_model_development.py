"""
01_logistic_model_development.py

Logistic regression model development for 7-day rebleeding in ANVUGIB.

This script belongs to a single-center retrospective cohort study of acute
non-variceal upper gastrointestinal bleeding (ANVUGIB) conducted at the
General Hospital of Central Theater Command between January 2020 and
August 2024, as described in the Materials and Methods.

Methodological alignment
------------------------
This script is designed to be fully consistent with the published Methods:

- Study design and population:
  A retrospective cohort of ANVUGIB patients (Sections 1–3).

- Predictors and data collection (Section 4):
  All candidate predictors (demographics, comorbidities, symptoms, vital
  signs, laboratory tests, and risk scores) were collected from admission
  to the index endoscopy.

- Missing data handling (Section 4.3):
  <5% missingness: single imputation (median for continuous, mode for
  categorical variables).
  ≥5% missingness: multiple imputation by chained equations (MICE) with
  10 imputations.
  These steps were performed in FreeStatistics. The CSV used here is
  assumed to be the FINAL analysis dataset after this procedure.

- Outliers and standardization (Section 4.4):
  Outliers were checked using boxplots and retained if clinically
  reasonable; continuous variables were standardized using Z-scores.
  These steps were also performed in FreeStatistics before exporting the
  final dataset.

- Variable selection (Section 5.2):
  LASSO regression with 10-fold cross-validation was used in
  FreeStatistics to identify the final set of predictors. This script
  uses that LASSO-selected predictor set, explicitly listed in
  PREDICTOR_COLUMNS below. No additional variable selection is performed
  in Python.

- Model type (Section 5.1):
  A multivariable logistic regression model is fitted to predict 7-day
  rebleeding (binary outcome).

Important:
- The file data/anvugib_final_analysis.csv is treated as the FINAL analysis
  dataset after:
  * missing data handling (single imputation and MICE),
  * outlier checking,
  * Z-score standardization of continuous predictors,
  * LASSO variable selection,
  all of which were executed in FreeStatistics, as specified in the
  Materials and Methods.
- This script does NOT re-implement MICE, standardization, or LASSO; it
  starts from the preprocessed dataset to refit the logistic regression
  model and reproduce regression coefficients and ORs.
- No real patient-level data are embedded in this code; it only operates
  on a local CSV file path under the user's control.
"""

import pathlib
from typing import List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

RANDOM_STATE = 42

# FINAL analysis dataset exported from FreeStatistics
DATA_PATH = pathlib.Path("data") / "anvugib_final_analysis.csv"

RESULTS_DIR = pathlib.Path("results")
MODELS_DIR = pathlib.Path("models")

# Outcome column (0/1) for 7-day rebleeding
OUTCOME_COLUMN = "rebleeding_7d"

# Predictor columns:
# IMPORTANT:
# - This list must be set to the exact set of predictors that were retained
#   after LASSO regression in FreeStatistics (Section 5.2).
# - Column names MUST match the columns in data/anvugib_final_analysis.csv.
PREDICTOR_COLUMNS: List[str] = [
    "syncope",
    "pulse",
    "bowel_sound",
    "RDW",
    "ALB",
    # add any additional LASSO-selected predictors here
]


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def load_data(path: pathlib.Path) -> pd.DataFrame:
    """
    Load the final analysis dataset for ANVUGIB rebleeding prediction.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file exported from FreeStatistics. This file is
        assumed to reflect all preprocessing steps described in Sections
        4.3 (missing data), 4.4 (standardization), and 5.2 (LASSO) of the
        Materials and Methods.

    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the final analysis dataset (2020–2023 cohort).
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for the preprocessing pipeline described in the Methods.

    Methodological note
    -------------------
    According to Section 4.3 and 4.4 of the Materials and Methods:
    - Missing data were handled under a missing-at-random assumption:
      * for predictors with <5% missingness: single imputation using
        the median (continuous) or mode (categorical);
      * for predictors with ≥5% missingness: multiple imputation by
        chained equations (MICE) with 10 imputations.
    - Outliers were inspected using boxplots and retained if within
      clinically reasonable ranges.
    - Continuous variables were standardized using Z-scores.

    In this study, these steps were performed in FreeStatistics before
    exporting the final analysis dataset (data/anvugib_final_analysis.csv).
    Therefore, this function is intentionally implemented as a no-op
    (pass-through) to keep the code flow aligned with the Methods while
    avoiding double preprocessing.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame representing the final analysis dataset.

    Returns
    -------
    df : pandas.DataFrame
        The same data frame, returned unchanged.
    """
    return df


def split_cohorts(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = RANDOM_STATE,
):
    """
    Split the 2020–2023 cohort into training and internal validation sets.

    The split is performed with a 7:3 ratio (Section 2.3
    and Section 6.1).

    Parameters
    ----------
    df : pandas.DataFrame
        Final analysis dataset (2020–2023 cohort) after all preprocessing.
    test_size : float, optional
        Proportion of data assigned to the internal validation set
        (default 0.3, corresponding to a 7:3 split).
    random_state : int, optional
        Random seed used for reproducible splitting.

    Returns
    -------
    X_train : pandas.DataFrame
        Predictor matrix for the training cohort.
    X_val : pandas.DataFrame
        Predictor matrix for the internal validation cohort.
    y_train : pandas.Series
        Outcome vector for the training cohort.
    y_val : pandas.Series
        Outcome vector for the internal validation cohort.
    """
    X = df[PREDICTOR_COLUMNS].copy()
    y = df[OUTCOME_COLUMN].astype(int).copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_val, y_train, y_val


def fit_logistic_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Fit a multivariable logistic regression model on the training cohort.

    The implementation uses scikit-learn's LogisticRegression without
    penalization (penalty='none'), consistent with a standard maximum
    likelihood logistic regression model as described in Section 5.1.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Predictor matrix for the training cohort.
    y_train : pandas.Series
        Binary outcome vector (0/1) for the training cohort.

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression
        Fitted logistic regression model.
    """
    model = LogisticRegression(
        penalty="none",
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X_train, y_train)
    return model


def extract_sklearn_coefficients(
    model: LogisticRegression,
    predictor_names: List[str],
) -> pd.DataFrame:
    """
    Extract logistic regression coefficients and intercept from a scikit-learn model.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Fitted logistic regression model.
    predictor_names : list of str
        Names of predictor variables in the same order as in the training matrix.

    Returns
    -------
    coef_df : pandas.DataFrame
        Tidy table containing the intercept and variable-level coefficients.
    """
    coef = model.coef_.flatten()
    intercept = model.intercept_[0]

    rows = [{"variable": "intercept", "coef": intercept}]
    rows.extend(
        {"variable": name, "coef": float(c)}
        for name, c in zip(predictor_names, coef)
    )
    coef_df = pd.DataFrame(rows)
    return coef_df


def compute_odds_ratios_with_ci(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Compute odds ratios and 95% confidence intervals using statsmodels.

    This function refits a logistic regression model using statsmodels.api.Logit
    on the same training cohort in order to obtain standard errors and
    confidence intervals for the regression coefficients, consistent with
    Section 7.1 (AUC with 95% CI) and the general practice of reporting
    ORs with 95% CI.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Predictor matrix for the training cohort.
    y_train : pandas.Series
        Binary outcome vector (0/1) for the training cohort.

    Returns
    -------
    or_df : pandas.DataFrame
        Data frame with one row per model term (including intercept), containing:
        - variable: term name (intercept or predictor),
        - coef: logistic regression coefficient (log-odds),
        - odds_ratio: exponentiated coefficient,
        - ci_lower_95, ci_upper_95: 95% confidence interval for the OR.
    """
    X_sm = sm.add_constant(X_train, has_constant="add")
    logit_model = sm.Logit(y_train, X_sm)
    result = logit_model.fit(disp=False)

    params = result.params
    conf_int = result.conf_int(alpha=0.05)

    or_values = np.exp(params)
    or_ci_lower = np.exp(conf_int[0])
    or_ci_upper = np.exp(conf_int[1])

    or_df = pd.DataFrame(
        {
            "variable": params.index,
            "coef": params.values,
            "odds_ratio": or_values.values,
            "ci_lower_95": or_ci_lower.values,
            "ci_upper_95": or_ci_upper.values,
        }
    )
    return or_df


def save_model(model: LogisticRegression, path: pathlib.Path) -> None:
    """
    Save the fitted scikit-learn model to disk using joblib.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Fitted logistic regression model.
    path : pathlib.Path
        Destination path for the serialized model file (.joblib).
    """
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for logistic model development.

    Workflow (matching the Methods)
    -------------------------------
    1) Load the fully preprocessed FINAL analysis dataset
       (data/anvugib_final_analysis.csv) corresponding to the 2020–2023
       ANVUGIB cohort after:
       - missing data handling and multiple imputation (Section 4.3),
       - outlier checking and Z-score standardization (Section 4.4),
       - LASSO variable selection (Section 5.2).
       These steps were performed in FreeStatistics.

    2) Apply a placeholder preprocessing function `preprocess_data()` which
       is a no-op in Python but documents the above steps explicitly.

    3) Split the data into a 7:3 training and internal validation set using
       stratification by the 7-day rebleeding outcome (Section 6.1).

    4) Fit a multivariable logistic regression model to the training cohort
       (Section 5.1).

    5) Extract and save:
       - regression coefficients (intercept and beta coefficients);
       - odds ratios and 95% confidence intervals derived via statsmodels.

    Outputs
    -------
    - results/logistic_coefficients.csv
    - results/logistic_odds_ratios_ci.csv
    - models/logistic_model.joblib

    No patient-level data are stored in this script or in the repository;
    CSV files remain under the investigator's control.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    df = preprocess_data(df)  # document Methods 4.3 and 4.4 (FreeStatistics)

    X_train, X_val, y_train, y_val = split_cohorts(df)

    model = fit_logistic_model(X_train, y_train)

    coef_df = extract_sklearn_coefficients(model, PREDICTOR_COLUMNS)
    coef_path = RESULTS_DIR / "logistic_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    or_df = compute_odds_ratios_with_ci(X_train, y_train)
    or_path = RESULTS_DIR / "logistic_odds_ratios_ci.csv"
    or_df.to_csv(or_path, index=False)

    model_path = MODELS_DIR / "logistic_model.joblib"
    save_model(model, model_path)


if __name__ == "__main__":
    main()
