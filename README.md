# ANVUGIB 7-day Rebleeding Risk Prediction Model

This repository accompanies the manuscript on the development, validation,
and web deployment of a 7-day rebleeding risk prediction model for
acute non-variceal upper gastrointestinal bleeding (ANVUGIB).

It provides:

- Python scripts related to model development, internal validation, and figure generation;
- A detailed workflow document in Chinese (`工作流程.md`);
- Basic instructions for local deployment of the Shiny application;
- Environment and dependency information (`requirements.txt`, `environment.yml`).

No raw patient-level data are included in this repository.

> **Data availability**
>
> In accordance with the approval of the Ethics Committee of the General
> Hospital of Central Theater Command and institutional regulations, this
> repository does **not** contain any raw or de-identified patient-level
> data. All scripts and documents are intended to make the analysis steps
> transparent and reproducible for users who have appropriate access to
> the underlying electronic medical record (EMR) data.

---

## 1. Analysis workflow

The overall workflow for data preprocessing, model development, validation
and deployment is described in detail in:

- `工作流程.md` – Workflow for Data Preprocessing, Model Development,
  Validation and Deployment (Chinese).

This document follows exactly the procedures reported in the “Materials
and Methods” section of the manuscript, including:

- study design, setting, and cohort construction;
- outcome definition (7-day rebleeding);
- candidate predictors and data collection timeline;
- handling of missing data and outliers, and Z-score standardisation of
  continuous variables;
- LASSO variable selection and final multivariable logistic regression;
- internal validation, temporal validation, and stratified analyses;
- model performance evaluation (discrimination, calibration, and decision
  curve analysis);
- deployment of the prediction model as a Shiny web application.

As stated in the manuscript, **all primary analyses were performed using
FreeStatistics software (V2.2.0)**. The Python scripts provided here are
designed to mirror the same analytical steps for transparency and do not
replace the FreeStatistics-based workflow that underlies the published
results.

---

## 2. Python scripts

The Python scripts assume that the user has access to a locally stored
analysis dataset corresponding to the 2020–2023 cohort after missing data
handling, standardisation and LASSO variable selection in FreeStatistics.
No dataset is distributed with this repository.

- `00_preprocessing_and_lasso.py` (optional example)  
  Demonstrates how the preprocessing and LASSO steps described in the
  Materials and Methods (single imputation, MICE with 10 imputations,
  Z-score standardisation, and LASSO regression with 10-fold
  cross-validation) could be implemented in Python using scikit-learn.  
  In the actual study, these steps were carried out in FreeStatistics,
  and the FreeStatistics-based results reported in the manuscript remain
  the authoritative analysis.

- `01_logistic_model_development.py`  
  Treats the exported analysis dataset (for example,
  `anvugib_final_analysis.csv`, held locally by the investigator) as the
  final dataset after preprocessing and LASSO in FreeStatistics. Using
  the set of predictors retained by LASSO, it fits a multivariable
  logistic regression model for 7-day rebleeding and outputs regression
  coefficients and odds ratios with 95% confidence intervals.

- `02_model_validation.py`  
  Implements the internal validation procedures described in the
  Materials and Methods. The script:
  - reproduces the 7:3 stratified split of the 2020–2023 cohort into
    training and internal validation sets;
  - evaluates discrimination using ROC curves and AUC with 95% confidence
    intervals (for example via bootstrap resampling and cross-validation);
  - evaluates calibration using calibration-in-the-large, calibration
    slope, and grouped calibration curves;
  - performs decision curve analysis (DCA) to quantify net benefit across
    a range of risk thresholds.  
  All outputs are summary-level and do not contain patient identifiers.

- `03_generate_figures.py`  
  Uses only the aggregated prediction outputs and DCA tables saved in a
  local `results/` directory (for example `predictions_internal.csv`,
  `dca_internal.csv`) to generate ROC, calibration, and DCA figures.  
  The script does not load raw patient-level data.

All Python scripts are intended to be run locally by investigators who
have appropriate access to the underlying data.

---

## 3. Shiny application: local deployment

The study model was implemented in an interactive Shiny web application.
The Shiny code can be placed in a directory such as `shiny_app/`
(containing `app.R` and related files) and run locally without uploading
any patient-level data to external servers.

> **Intended use**
>
> The Shiny application is intended for research and educational use only
> and is **not** a production-grade clinical decision support system.

### 3.1 Prerequisites

- R (version 4.5.1 was used in the original analysis; R ≥ 4.3.0 should
  also work);
- RStudio (optional but recommended);
- R packages:
  - `shiny`
  - `dplyr`
  - `ggplot2`
  - `DT`
  - `readr`

All required R packages can be installed by running the helper script
`安装 R 包` in R, for example:

    source("安装 R 包")

(If an English filename is preferred, the script can be renamed to
`install_packages.R` and sourced accordingly.)

### 3.2 How to run the Shiny app locally

Assuming the Shiny application files are stored under a directory such as
`shiny_app/` in this repository:

1. Clone this repository to your local machine:

       git clone https://github.com/<your-username>/<your-repo>.git
       cd <your-repo>

2. Open R or RStudio and set the working directory to the `shiny_app/`
   folder, for example:

       setwd("path/to/your-repo/shiny_app")

3. Install all required R packages (only needed for the first run):

       source("../安装 R 包")

4. Start the Shiny application:

       shiny::runApp(".")

5. The application will open in the default web browser at an address
   similar to `http://127.0.0.1:xxxx`.

### 3.3 Note on cloud deployment

In the manuscript, the web-based version of the model is deployed on a
cloud Shiny hosting service (for example shinyapps.io). The same R/Shiny
code can be used for either:

- local deployment (as described above), or  
- cloud deployment on a Shiny hosting platform.

Cloud deployment uses the same Shiny application code but a different
hosting infrastructure, as discussed in the Limitations and Shiny hosting
sections of the manuscript.

---

## 4. Environment information and dependencies

### 4.1 Software versions

The following software versions were used in the original analysis:

- Operating system: Windows 10 / 11 (64-bit);
- R: 4.5.1;
- Python: 3.9.13;
- FreeStatistics: V2.2.0 (primary analysis environment, as described in
  the manuscript).

### 4.2 R dependencies (for the Shiny app)

The Shiny application depends on the following R packages:

- `shiny`
- `dplyr`
- `ggplot2`
- `DT`
- `readr`

These packages can be installed using:

    source("安装 R 包")

### 4.3 Python dependencies and environment configuration

The Python scripts were developed and tested with the following Python
packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `statsmodels`
- `matplotlib`
- `joblib`

A minimal `requirements.txt` file is provided to allow users to recreate
the Python environment:

    pip install -r requirements.txt

Alternatively, a Conda environment can be created using
`environment.yml`:

    conda env create -f environment.yml
    conda activate anvugib-rebleeding

Both `requirements.txt` and `environment.yml` are included in this
repository and specify the package versions used in the original analysis.

---

## 5. Repository structure

The main files in the repository are organised as follows:

    .
    ├─ 00_preprocessing_and_lasso.py      # optional example for preprocessing + LASSO
    ├─ 01_logistic_model_development.py   # logistic regression model fitting
    ├─ 02_model_validation.py             # internal validation, calibration and DCA
    ├─ 03_generate_figures.py             # plotting ROC, calibration and DCA curves
    ├─ 工作流程.md                         # detailed workflow description (Chinese)
    ├─ environment.yml                    # Conda environment specification
    ├─ 安装 R 包                           # helper script to install required R packages
    ├─ requirements.txt                   # Python package list (pip)
    └─ README.md

The following folders are intended for local use only and are not
required in the public repository:

- `data/` – local copies of analysis datasets (not included here);
- `results/` – model outputs generated by the Python scripts (for example
  prediction tables, DCA tables);
- `figures/` – plots generated by the Python scripts;
- `models/` – serialized model objects (for example `.joblib`).

These directories are mentioned to clarify how the scripts are used
locally, but they do not contain any patient-level data in the public
repository.
