"""
03_generate_figures.py

Generate ROC, calibration, and decision curve analysis (DCA) figures for the
ANVUGIB 7-day rebleeding prediction model.

Methodological alignment
------------------------
This script corresponds to the evaluation section of the Materials and Methods:

- Discrimination (Section 7.1):
  ROC curves and AUCs are visualized for the internal (and optionally
  temporal) validation cohorts.

- Calibration (Section 7.2):
  Calibration curves are drawn based on grouped averages of predicted
  probabilities vs. observed 7-day rebleeding rates.

- Clinical utility (Section 7.3):
  Decision curve analysis (DCA) curves are plotted to show net benefit
  across a range of risk thresholds.

Preprocessing and model fitting (Sections 4–6) are implemented and
documented in Scripts 01 and 02. All missing data handling, multiple
imputation, standardization, and LASSO steps were executed in
FreeStatistics before exporting the final analysis dataset.

This script uses ONLY the aggregated prediction outputs and DCA tables
saved in the results/ directory (e.g., predictions_internal.csv,
dca_internal.csv). It does not access raw patient-level data.
"""

import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

RESULTS_DIR = pathlib.Path("results")
FIGURES_DIR = pathlib.Path("figures")


# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------

def load_predictions(path: pathlib.Path) -> Optional[pd.DataFrame]:
    """
    Load prediction results (true outcome and predicted probabilities).

    Parameters
    ----------
    path : pathlib.Path
        CSV file containing at least two columns:
        - "y_true": observed binary outcome (0/1),
        - "y_pred": predicted probability of 7-day rebleeding.

    Returns
    -------
    df : pandas.DataFrame or None
        Loaded data frame, or None if the file does not exist.
    """
    if not path.exists():
        print(f"[INFO] Prediction file not found, skipping: {path}")
        return None
    df = pd.read_csv(path)
    return df


def load_dca_results(path: pathlib.Path) -> Optional[pd.DataFrame]:
    """
    Load decision curve analysis (DCA) results from CSV.

    Parameters
    ----------
    path : pathlib.Path
        CSV file created by Script 2 (e.g., 'results/dca_internal.csv').

    Returns
    -------
    df : pandas.DataFrame or None
        Loaded DCA results, or None if the file does not exist.
    """
    if not path.exists():
        print(f"[INFO] DCA file not found, skipping: {path}")
        return None
    df = pd.read_csv(path)
    return df


# -------------------------------------------------------------------------
# Plotting functions
# -------------------------------------------------------------------------

def plot_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    out_path: pathlib.Path,
) -> None:
    """
    Plot an ROC curve for a given dataset and save it to disk.

    The AUC is computed on the fly and included in the legend for reference.
    No patient-level data are exposed.

    Parameters
    ----------
    y_true : array-like
        Binary outcome vector.
    y_pred : array-like
        Predicted probabilities for the positive class.
    label : str
        Text label indicating the dataset (e.g. "Internal validation").
    out_path : pathlib.Path
        Output file path for the figure (.png or .pdf).
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_value = roc_auc_score(y_true, y_pred)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve
