"""
00_preprocessing_and_lasso.py

Preprocessing and LASSO variable selection for the 7-day rebleeding model
in acute non-variceal upper gastrointestinal bleeding (ANVUGIB).

This script provides a Python implementation of the preprocessing steps
described in the Materials and Methods:

- Missing data handling (Section 4.3):
  <5% missingness: single imputation (median for continuous variables,
  mode for categorical variables);
  ≥5% missingness: multiple imputation by chained equations (MICE) with
  10 imputations.

- Outliers and standardisation (Section 4.4):
  Outliers are inspected and retained if clinically reasonable;
  continuous variables are standardised using Z-scores.

- Variable selection (Section 5.2):
  LASSO regression with 10-fold cross-validation is used to identify
  candidate predictors for the final multivariable logistic regression model.

Important notes
---------------
- In the primary analysis for the manuscript, ALL of these steps
  (single imputation, MICE, Z-score standardisation, and LASSO variable
  selection) were performed in FreeStatistics (V2.2.0), as stated in the
  Materials and Methods. The results reported in the paper are based on
  the FreeStatistics workflow and should be regarded as the authoritative
  analysis.

- This Python script is provided as an OPTIONAL / EXAMPLE implementation
  for users who wish to reproduce the workflow starting from a raw CSV.
  Because implementation details (e.g. MICE algorithm, optimisation,
  penalty tuning) may differ between FreeStatistics and scikit-learn,
  the set of selected predictors and corresponding coefficients obtained
  from this script may NOT be identical to those in the manuscript.

- This script assumes that the user has a local raw dataset
  (data/anvugib_raw.csv) with all candidate predictors and the outcome
  variable 'rebleeding_7d'. No patient-level data are embedded in this
  code or repository.

- The imputed and standardised dataset exported by this script is saved
  as data/anvugib_final_analysis.csv, which can be used directly by
  01_logistic_model_development.py and 02_model_validation.py.
"""

import pathlib
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

RANDOM_STATE = 42

RAW_DATA_PATH = pathlib.Path("data") / "anvugib_raw.csv"
FINAL_DATA_PATH = pathlib.Path("data") / "anvugib_final_analysis.csv"

RESULTS_DIR = pathlib.Path("results")

# NOTE:
# ----
# This script demonstrates how the preprocessing and LASSO steps described
# in the Materials and Methods COULD be implemented in Python.
# In the actual study, the corresponding analyses were carried out in
# FreeStatistics (V2.2.0); therefore, the FreeStatistics-based results
# reported in the manuscript prevail in case of any discrepancies.

# Outcome variable: 7-day rebleeding (0/1)
OUTCOME_COLUMN = "rebleeding_7d"

# Candidate predictors:
# --------------------------------------------------------------
# IMPORTANT:
# - This list should include ALL candidate predictors described
#   in Section 4.1 of the Materials and Methods.
# - Column names MUST match the names in data/anvugib_raw.csv.
# - Please fill in according to your real dataset.
CANDIDATE_PREDICTORS: List[str] = [
    # Demographics
    "age",
    "sex",
    "smoking",
    "alcohol_use",
    # Comorbidities
    "hypertension",
    "diabetes",
    "coronary_artery_disease",
    "hepatic_dysfunction",
    "chronic_kidney_disease",
    # Medications
    "anticoagulants",
    "antiplatelets",
    "NSAIDs",
    # Clinical symptoms
    "hematemesis",
    "melena",
    "syncope",
    # Vital signs
    "SBP",
    "pulse",
    "bowel_sound",
    # Laboratory tests
    "hemoglobin",
    "platelet_count",
    "albumin",
    "creatinine",
    "INR",
    "PT",
    # Risk scores
    "GBS",
    "AIMS65",
    # If additional candidates exist, add them here
]

# Continuous variables (for median imputation + Z-score standardisation).
# These should be a subset of CANDIDATE_PREDICTORS.
CONTINUOUS_VARS: List[str] = [
    "age",
    "SBP",
    "pulse",
    "hemoglobin",
    "platelet_count",
    "albumin",
    "creatinine",
    "INR",
    "PT",
    "GBS",
    "AIMS65",
    # Add other continuous predictors here if applicable (e.g. RDW)
]

# All remaining candidate predictors are treated as categorical.
# We will infer categorical variables as those in CANDIDATE_PREDICTORS
# that are not in CONTINUOUS_VARS.
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------


def load_raw_data(path: pathlib.Path) -> pd.DataFrame:
    """
    Load the raw ANVUGIB dataset from CSV.

    Parameters
    ----------
    path : pathlib.Path
        Path to data/anvugib_raw.csv.

    Returns
    -------
    df : pandas.DataFrame
        Raw dataset containing candidate predictors and outcome.
    """
    df = pd.read_csv(path)
    return df


def compute_missingness(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.Series:
    """
    Compute the proportion of missing values for each specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    columns : list of str
        Columns for which to compute missingness.

    Returns
    -------
    missing_pct : pandas.Series
        Series indexed by column name, with missingness proportion (0–1).
    """
    missing_pct = df[columns].isna().mean()
    return missing_pct


def split_by_missingness(
    missing_pct: pd.Series,
    threshold: float = 0.05,
) -> Tuple[List[str], List[str]]:
    """
    Split variables into low-missing and high-missing groups.

    Parameters
    ----------
    missing_pct : pandas.Series
        Series of missingness proportions indexed by column name.
    threshold : float, optional
        Threshold to distinguish low (< threshold) vs high (>= threshold)
        missingness. Default is 0.05 (5%).

    Returns
    -------
    low_missing : list of str
        Variables with missingness < threshold.
    high_missing : list of str
        Variables with missingness >= threshold.
    """
    low_missing = missing_pct[missing_pct < threshold].index.tolist()
    high_missing = missing_pct[missing_pct >= threshold].index.tolist()
    return low_missing, high_missing


def simple_imputation(
    df: pd.DataFrame,
    continuous_vars: List[str],
    categorical_vars: List[str],
    low_missing_vars: List[str],
) -> pd.DataFrame:
    """
    Perform single imputation for variables with <5% missingness.

    - Continuous variables: median.
    - Categorical variables: mode (most frequent category).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    continuous_vars : list of str
        Names of continuous variables.
    categorical_vars : list of str
        Names of categorical variables.
    low_missing_vars : list of str
        Variables with missingness < 5%.

    Returns
    -------
    df_imputed : pandas.DataFrame
        Dataset with low-missing variables imputed.
    """
    df_imputed = df.copy()

    for col in low_missing_vars:
        if col in continuous_vars:
            median_value = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_value)
        elif col in categorical_vars:
            mode_series = df_imputed[col].mode(dropna=True)
            if not mode_series.empty:
                mode_value = mode_series.iloc[0]
                df_imputed[col] = df_imputed[col].fillna(mode_value)
        # If a variable is neither in continuous nor categorical lists
        # (should not happen), leave it unchanged.
    return df_imputed


def run_mice_imputations(
    df: pd.DataFrame,
    variables_for_mice: List[str],
    n_imputations: int = 10,
    random_state: int = RANDOM_STATE,
) -> Dict[int, pd.DataFrame]:
    """
    Run multiple imputation by chained equations (MICE) for variables
    with ≥5% missingness.

    Implementation details
    ----------------------
    - This function uses sklearn.impute.IterativeImputer as a practical
      implementation of MICE.
    - To obtain 10 imputed datasets, the imputation procedure is repeated
      10 times with different random seeds:
        random_state + 0, random_state + 1, ..., random_state + 9.
    - All candidate predictors (including those with low missingness)
      can be included as predictors in the imputation model; however,
      only variables listed in 'variables_for_mice' will have their
      missing values updated.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset after simple imputation for low-missing variables.
    variables_for_mice : list of str
        Variables with ≥5% missingness to be imputed using MICE.
    n_imputations : int, optional
        Number of imputed datasets to generate (default 10).
    random_state : int, optional
        Base random seed.

    Returns
    -------
    imputed_datasets : dict
        Dictionary mapping imputation index (1..n_imputations) to
        imputed pandas.DataFrame.
    """
    imputed_datasets: Dict[int, pd.DataFrame] = {}

    if not variables_for_mice:
        # No high-missingness variables; nothing to impute via MICE.
        imputed_datasets[1] = df.copy()
        return imputed_datasets

    # For MICE, we work on a numeric copy of the variables. It is assumed
    # that categorical predictors are already encoded numerically
    # (e.g. 0/1 indicators) in the raw dataset, consistent with the
    # preprocessing performed in FreeStatistics.
    mice_input = df[variables_for_mice].copy()

    for m in range(n_imputations):
        seed = random_state + m
        imputer = IterativeImputer(
            random_state=seed,
            sample_posterior=True,
            max_iter=10,
        )
        imputed_array = imputer.fit_transform(mice_input)
        df_imputed = df.copy()
        df_imputed[variables_for_mice] = imputed_array
        imputed_datasets[m + 1] = df_imputed

    return imputed_datasets


def standardise_continuous(
    df: pd.DataFrame,
    continuous_vars: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Standardise continuous variables using Z-scores.

    For each continuous variable x, the Z-score is defined as:

        z = (x - mean) / std,

    where mean and std are computed from the current dataset,
    consistent with Section 4.4.

    Parameters
    ----------
    df : pandas.DataFrame
        Imputed dataset.
    continuous_vars : list of str
        Continuous variables to standardise.

    Returns
    -------
    df_std : pandas.DataFrame
        Dataset with standardised continuous variables (original variables
        overwritten by their Z-scores).
    means : pandas.Series
        Means used for standardisation.
    stds : pandas.Series
        Standard deviations used for standardisation.
    """
    df_std = df.copy()
    scaler = StandardScaler()
    X_cont = df_std[continuous_vars].values

    X_std = scaler.fit_transform(X_cont)
    df_std[continuous_vars] = X_std

    means = pd.Series(
        scaler.mean_,
        index=continuous_vars,
        name="mean",
    )
    stds = pd.Series(
        np.sqrt(scaler.var_),
        index=continuous_vars,
        name="std",
    )

    return df_std, means, stds


def run_lasso_selection(
    df: pd.DataFrame,
    predictors: List[str],
    outcome_col: str = OUTCOME_COLUMN,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Perform LASSO variable selection using logistic regression with L1 penalty.

    A LogisticRegressionCV model with L1 penalty and 10-fold cross-validation
    is fitted on the standardised predictors to select variables with
    non-zero coefficients at the optimal regularisation strength.

    This corresponds to the LASSO procedure described in Section 5.2
    (LASSO regression with 10-fold cross-validation).

    Parameters
    ----------
    df : pandas.DataFrame
        Imputed and standardised dataset.
    predictors : list of str
        Names of predictor variables to be considered in LASSO.
    outcome_col : str, optional
        Outcome column name (default 'rebleeding_7d').
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    selected_predictors : list of str
        Predictors with non-zero coefficients at the optimal C.
    coef_table : pandas.DataFrame
        Table with one row per predictor containing:
        - predictor: predictor name,
        - coef: coefficient estimate at optimal C.
    """
    X = df[predictors].values
    y = df[outcome_col].astype(int).values

    # L1-penalised logistic regression with 10-fold CV
    lasso_cv = LogisticRegressionCV(
        Cs=10,
        cv=10,
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        refit=True,
        random_state=random_state,
        max_iter=1000,
    )
    lasso_cv.fit(X, y)

    coefs = lasso_cv.coef_.ravel()
    coef_table = pd.DataFrame(
        {
            "predictor": predictors,
            "coef": coefs,
        }
    )

    selected_predictors = coef_table.loc[coef_table["coef"] != 0, "predictor"].tolist()
    return selected_predictors, coef_table


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    """
    Main entry point for preprocessing and LASSO variable selection.

    Workflow (matching Sections 4 and 5)
    ------------------------------------
    1) Load the raw dataset (data/anvugib_raw.csv), containing all
       candidate predictors (Section 4.1) and the binary outcome
       'rebleeding_7d' (Section 3.1).

    2) Compute missingness proportions for all candidate predictors and
       split them into:
       - low-missing (<5%) variables,
       - high-missing (≥5%) variables, consistent with Section 4.3.

    3) Apply single imputation for low-missing variables:
       - continuous: median,
       - categorical: mode.

    4) Apply MICE (IterativeImputer) to variables with ≥5% missingness,
       generating 10 imputed datasets (Section 4.3, 10 imputations).

    5) For each imputed dataset, standardise continuous variables using
       Z-scores (Section 4.4).

    6) On the first imputed + standardised dataset (imputation #1),
       perform LASSO logistic regression with 10-fold cross-validation to
       select predictors (Section 5.2).

    7) Save:
       - one final analysis dataset as data/anvugib_final_analysis.csv,
         which can be used by Scripts 01 and 02;
       - the LASSO coefficient table and selected predictors list as
         CSV files in results/.

    Note
    ----
    - In the published study, the corresponding preprocessing and LASSO
      steps were carried out in FreeStatistics (V2.2.0). This Python
      script is intended for didactic and reproducibility support only.
      In case of any discrepancies, the FreeStatistics-based results
      reported in the manuscript should be used.

    - The set of selected predictors can be copied into Script 01 as
      PREDICTOR_COLUMNS to ensure consistency.
    - Imputed datasets generated by this script are intended for local
      use only and are not expected to be committed to a public repository.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Determine categorical variables as the complement of CONTINUOUS_VARS
    categorical_vars = [
        col for col in CANDIDATE_PREDICTORS if col not in CONTINUOUS_VARS
    ]

    # 1) Load raw data
    df_raw = load_raw_data(RAW_DATA_PATH)

    # 2) Compute missingness among candidate predictors
    missing_pct = compute_missingness(df_raw, CANDIDATE_PREDICTORS)
    low_missing, high_missing = split_by_missingness(missing_pct, threshold=0.05)

    # 3) Single imputation for low-missing variables
    df_after_simple = simple_imputation(
        df=df_raw,
        continuous_vars=CONTINUOUS_VARS,
        categorical_vars=categorical_vars,
        low_missing_vars=low_missing,
    )

    # 4) MICE for high-missing variables (10 imputations)
    imputed_datasets = run_mice_imputations(
        df=df_after_simple,
        variables_for_mice=high_missing,
        n_imputations=10,
        random_state=RANDOM_STATE,
    )

    # 5) Standardisation and 6) LASSO on imputation #1
    df_imp1 = imputed_datasets[1].copy()

    df_imp1_std, means, stds = standardise_continuous(
        df=df_imp1,
        continuous_vars=CONTINUOUS_VARS,
    )

    # 6) LASSO selection
    selected_predictors, coef_table = run_lasso_selection(
        df=df_imp1_std,
        predictors=CANDIDATE_PREDICTORS,
        outcome_col=OUTCOME_COLUMN,
        random_state=RANDOM_STATE,
    )

    # Save LASSO results
    coef_table.to_csv(
        RESULTS_DIR / "lasso_coefficients.csv",
        index=False,
    )
    pd.DataFrame({"predictor": selected_predictors}).to_csv(
        RESULTS_DIR / "lasso_selected_predictors.csv",
        index=False,
    )

    # 7) Save the final analysis dataset for use in Scripts 01 and 02
    #    Only candidates + outcome are needed for modelling.
    final_cols = CANDIDATE_PREDICTORS + [OUTCOME_COLUMN]
    df_final = df_imp1_std[final_cols].copy()
    df_final.to_csv(FINAL_DATA_PATH, index=False)

    # No AUC or OR values are printed or stored here; this script only
    # prepares the final analysis dataset and LASSO variable selection
    # results, consistent with the Methods.


if __name__ == "__main__":
    main()
