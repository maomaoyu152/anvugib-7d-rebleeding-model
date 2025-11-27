"""
02_model_validation.py

Internal validation of the logistic regression model for 7-day rebleeding
in ANVUGIB (single-center retrospective cohort).

Methodological alignment
------------------------
This script implements the internal validation procedures described in the
Materials and Methods:

- Internal validation cohort (Section 6.1):
  The 2020–2023 cohort is randomly split into a 7:3 training and internal
  validation set, stratified by the 7-day rebleeding outcome.

- Discrimination (Section 7.1):
  Discrimination is evaluated using ROC curves and AUCs with 95% confidence
  intervals. In this script, bootstrap resampling and 5-fold cross-validation
  are used as practical implementations to obtain robust AUC estimates,
  consistent with the description "AUCs and 95% confidence intervals".

- Calibration (Section 7.2):
  Calibration-in-the-large (intercept) and calibration slope are computed
  by regressing the observed outcome on the logit of the predicted
  probabilities. Grouped calibration curves are also produced.

- Clinical utility (Section 7.3):
  Decision curve analysis (DCA) is performed to calculate net benefit
  across a range of risk thresholds.

Preprocessing note
------------------
All upstream preprocessing steps described in Sections 4.3 and 4.4
(missing data handling with single imputation and MICE, outlier checks,
Z-score standardization) as well as LASSO variable selection (Section 5.2)
were performed in FreeStatistics before exporting the final analysis
dataset (data/anvugib_final_analysis.csv). This script therefore assumes
that the input CSV is the FINAL, fully preprocessed dataset and does not
re-run these steps.

No real patient-level data are embedded in this script; it operates only
on a local CSV file path under the investigator's control.
"""

import pathlib
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

RANDOM_STATE = 42
DATA_PATH = pathlib.Path("data") / "anvugib_final_analysis.csv"
RESULTS_DIR = pathlib.Path("results")
MODELS_DIR = pathlib.Path("models")

OUTCOME_COLUMN = "rebleeding_7d"

# Must match Script 1 (LASSO-selected predictors)
PREDICTOR_COLUMNS = [
    "syncope",
    "pulse",
    "bowel_sound",
    "RDW",
    "ALB",
    # add any additional predictors here in the correct order
]


# -------------------------------------------------------------------------
# Data loading, preprocessing (documented), and splitting
# -------------------------------------------------------------------------

def load_data(path: pathlib.Path) -> pd.DataFrame:
    """
    Load the final analysis dataset for ANVUGIB rebleeding prediction.

    The input CSV is assumed to be the fully preprocessed dataset after
    missing data handling, Z-score standardization, and LASSO variable
    selection performed in FreeStatistics.
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for the preprocessing steps already performed in FreeStatistics.

    Methodological note (Sections 4.3, 4.4, 5.2)
    -------------------------------------------
    - Missing data:
      * <5% missingness: single imputation (median/mode).
      * ≥5% missingness: multiple imputation by chained equations (MICE)
        with 10 imputations.
    - Outliers and standardization:
      Outliers were checked using boxplots and retained if clinically
      acceptable; continuous variables were standardized to Z-scores.
    - LASSO:
      LASSO regression with 10-fold cross-validation was used to select
      predictors for the final model.

    All of the above were executed in FreeStatistics before exporting
    data/anvugib_final_analysis.csv. To keep the code flow aligned with
    the Methods while avoiding double preprocessing, this function returns
    the input data frame unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        Final analysis dataset.

    Returns
    -------
    df : pandas.DataFrame
        Same dataset, unchanged.
    """
    return df


def split_cohorts(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the 2020–2023 cohort into training and internal validation sets.

    A 7:3 split is performed using stratified sampling on the 7-day
    rebleeding outcome, consistent with Section 6.1 of the Methods.
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


def load_or_fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Load the scikit-learn logistic regression model if available; otherwise,
    fit a new model using the same specification as in Script 1.

    This ensures consistency between model development and validation.
    """
    import joblib

    model_path = MODELS_DIR / "logistic_model.joblib"
    if model_path.exists():
        model: LogisticRegression = joblib.load(model_path)
        return model

    model = LogisticRegression(
        penalty="none",
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------------------
# AUC and cross-validation utilities
# -------------------------------------------------------------------------

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = RANDOM_STATE,
) -> Tuple[float, float, float]:
    """
    Estimate AUC and its 95% confidence interval using bootstrap resampling.

    This implements the requirement of reporting AUCs with 95% confidence
    intervals (Section 7.1) in a practical way.
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    auc_scores = []
    n_samples = y_true.shape[0]

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        auc_scores.append(score)

    auc_scores = np.array(auc_scores)
    auc_mean = roc_auc_score(y_true, y_pred)
    auc_ci_lower = np.percentile(auc_scores, 2.5)
    auc_ci_upper = np.percentile(auc_scores, 97.5)
    return auc_mean, auc_ci_lower, auc_ci_upper


def cross_validated_auc(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform K-fold (e.g., 5-fold) stratified cross-validation to obtain
    out-of-fold predicted probabilities and corresponding AUC.

"""
This function provides an additional internal validation view using
5-fold stratified cross-validation. Although cross-validation was not
reported in the main manuscript, it can be useful for sensitivity
analyses and for users who wish to explore the model's stability.
"""

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        model = LogisticRegression(
            penalty="none",
            solver="lbfgs",
            max_iter=1000,
        )
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict_proba(X_test_fold)[:, 1]

        y_true_all.append(y_test_fold.values)
        y_pred_all.append(y_pred_fold)

    y_true_cv = np.concatenate(y_true_all)
    y_pred_cv = np.concatenate(y_pred_all)
    return y_true_cv, y_pred_cv


# -------------------------------------------------------------------------
# Calibration and DCA
# -------------------------------------------------------------------------

def calibration_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute calibration-in-the-large (intercept) and calibration slope.

    A logistic regression model of the observed outcome on the logit of the
    predicted probabilities is fitted:

        logit(P(Y=1)) = alpha + beta * logit(p_hat),

    where alpha is the calibration intercept and beta is the calibration slope,
    consistent with Section 7.2.
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    logit_pred = np.log(y_pred / (1 - y_pred))

    X = sm.add_constant(logit_pred, has_constant="add")
    logit_model = sm.Logit(y_true, X)
    result = logit_model.fit(disp=False)

    intercept = float(result.params[0])
    slope = float(result.params[1])
    return intercept, slope


def decision_curve_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Perform decision curve analysis (DCA) for a single prediction model.

    For each risk threshold p_t, the net benefit is calculated as:

        NB_model = (TP / N) - (FP / N) * (p_t / (1 - p_t)),

    where TP and FP are the numbers of true and false positives when applying
    the threshold p_t.

    The net benefit of two reference strategies is also provided:
    - "treat all": NB_all,
    - "treat none": NB_none = 0.

    This implements the clinical utility evaluation described in Section 7.3.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = y_true.shape[0]

    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    event_rate = y_true.mean()
    dca_rows = []

    for p_t in thresholds:
        predicted_positive = y_pred >= p_t

        TP = np.logical_and(predicted_positive, y_true == 1).sum()
        FP = np.logical_and(predicted_positive, y_true == 0).sum()

        if N == 0:
            continue

        nb_model = (TP / N) - (FP / N) * (p_t / (1 - p_t))
        nb_all = event_rate - (1 - event_rate) * (p_t / (1 - p_t))
        nb_none = 0.0

        dca_rows.append(
            {
                "threshold": p_t,
                "net_benefit_model": nb_model,
                "net_benefit_all": nb_all,
                "net_benefit_none": nb_none,
            }
        )

    dca_df = pd.DataFrame(dca_rows)
    return dca_df


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for internal model validation.

    Workflow (matching Sections 6–7)
    --------------------------------
    1) Load the final analysis dataset (2020–2023 cohort) after all
       preprocessing and LASSO selection in FreeStatistics.

    2) Apply a placeholder preprocessing function that documents the
       upstream steps (4.3, 4.4, 5.2) but leaves the dataset unchanged.

    3) Reproduce the 7:3 training / internal validation split (Section 6.1).

    4) Load (or refit) the logistic regression model from Script 1
       (Section 5.1).

    5) Evaluate discrimination on the internal validation set:
       - ROC curve and AUC with 95% CI via bootstrap.
       - Optionally, 5-fold cross-validation AUC as an additional internal
         validation layer.

    6) Evaluate calibration on the internal validation set:
       - calibration intercept (calibration-in-the-large),
       - calibration slope,
       - grouped calibration curve.

    7) Perform decision curve analysis (DCA) and save net benefit across
       thresholds.

    Outputs
    -------
    - results/roc_internal.csv
    - results/predictions_internal.csv
    - results/predictions_cv5.csv  (5-fold CV, optional use)
    - results/calibration_stats_internal.csv
    - results/calibration_curve_internal.csv
    - results/dca_internal.csv

    All outputs are summary-level and derived from the final analysis
    dataset; no raw patient identifiers are included.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    df = preprocess_data(df)  # document Methods 4.3/4.4/5.2 (FreeStatistics)

    X_train, X_val, y_train, y_val = split_cohorts(df)

    model = load_or_fit_model(X_train, y_train)

    # --- Hold-out internal validation performance ---
    y_val_pred = model.predict_proba(X_val)[:, 1]
    auc_val, auc_val_lower, auc_val_upper = bootstrap_auc_ci(y_val, y_val_pred)

    # Save ROC curve points for the internal validation set
    fpr, tpr, roc_thresholds = roc_curve(y_val, y_val_pred)
    roc_df = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": roc_thresholds,
        }
    )
    roc_df.to_csv(RESULTS_DIR / "roc_internal.csv", index=False)

    # Save internal validation predictions for later plotting
    pred_internal_df = pd.DataFrame(
        {
            "y_true": y_val.values,
            "y_pred": y_val_pred,
        }
    )
    pred_internal_df.to_csv(
        RESULTS_DIR / "predictions_internal.csv",
        index=False,
    )

    # --- Cross-validated AUC on the full dataset (5-fold CV) ---
    y_true_cv, y_pred_cv = cross_validated_auc(
        df[PREDICTOR_COLUMNS],
        df[OUTCOME_COLUMN].astype(int),
        n_splits=5,
        random_state=RANDOM_STATE,
    )
    auc_cv, auc_cv_lower, auc_cv_upper = bootstrap_auc_ci(y_true_cv, y_pred_cv)

    cv_pred_df = pd.DataFrame(
        {
            "y_true": y_true_cv,
            "y_pred": y_pred_cv,
        }
    )
    cv_pred_df.to_csv(
        RESULTS_DIR / "predictions_cv5.csv",
        index=False,
    )

    # --- Calibration statistics on the internal validation set ---
    cal_intercept, cal_slope = calibration_statistics(y_val.values, y_val_pred)
    cal_stats_df = pd.DataFrame(
        [
            {
                "dataset": "internal_validation",
                "calibration_intercept": cal_intercept,
                "calibration_slope": cal_slope,
            }
        ]
    )
    cal_stats_df.to_csv(
        RESULTS_DIR / "calibration_stats_internal.csv",
        index=False,
    )

    # Grouped calibration curve (e.g., 10 quantile-based bins)
    n_bins = 10
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(y_val_pred, quantiles)

    bin_ids = np.digitize(y_val_pred, bin_edges[1:-1], right=True)
    calibration_rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_pred_mean = float(y_val_pred[mask].mean())
        bin_obs_rate = float(y_val.values[mask].mean())
        calibration_rows.append(
            {
                "bin": b,
                "predicted_mean": bin_pred_mean,
                "observed_rate": bin_obs_rate,
            }
        )
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(
        RESULTS_DIR / "calibration_curve_internal.csv",
        index=False,
    )

    # --- Decision curve analysis on the internal validation set ---
    dca_internal_df = decision_curve_analysis(y_val.values, y_val_pred)
    dca_internal_df.to_csv(
        RESULTS_DIR / "dca_internal.csv",
        index=False,
    )

    # Temporal validation cohort (Section 6.2) is handled separately and is
    # not included in this script because the corresponding dataset is not
    # provided in this repository. If needed, a similar pipeline can be
    # applied to that cohort and its DCA saved as results/dca_temporal.csv.
    # Script 04_generate_figures.py is prepared to plot both internal and
    # temporal validation curves if such files are present.


if __name__ == "__main__":
    main()
