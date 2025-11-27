# ANVUGIB 7-day Rebleeding Risk Prediction Model

This repository accompanies the manuscript on the development, validation,
and web deployment of a 7-day rebleeding risk prediction model for
acute non-variceal upper gastrointestinal bleeding (ANVUGIB). It includes:

- Python scripts to reproduce the main model development and internal validation;
- A Shiny web application for interactive risk calculation;
- Basic instructions for local deployment and environment configuration.

No raw patient-level data are included in this repository.

---

## 1. Shiny application: local deployment

This repository contains a Shiny web application that implements the
7-day rebleeding risk prediction model for ANVUGIB. The application can
be run locally without uploading any patient-level data to external servers.

### 1.1 Prerequisites

- R (version 4.5.1 was used in the original analysis; R ≥ 4.3.0 should also work)
- RStudio (optional but recommended)
- R packages:
  - `shiny`
  - `dplyr`
  - `ggplot2`
  - `DT`
  - `readr`

All required R packages can be installed by running the script
`install_packages.R` in R:

    source("install_packages.R")

### 1.2 Shiny application location

The Shiny application is stored in the `shiny_app/` directory:

    shiny_app/
      app.R
      global.R      # if applicable
      ui.R          # if applicable
      server.R      # if applicable

The main entry point is `app.R`. Depending on the implementation, the
application may be organized as a single-file Shiny app (`app.R` only)
or with separate `ui.R` and `server.R` files.

### 1.3 How to run the Shiny app locally

1. Clone this repository to your local machine:

       git clone https://github.com/<your-username>/<your-repo>.git
       cd <your-repo>

2. Open R or RStudio and set the working directory to the `shiny_app/`
   folder, for example:

       setwd("path/to/your-repo/shiny_app")

3. Install all required R packages (only needed for the first run):

       source("../install_packages.R")

4. Start the Shiny application:

       shiny::runApp(".")

5. The application will open automatically in your default web browser
   at an address similar to `http://127.0.0.1:xxxx`.

### 1.4 Note on cloud deployment

In the manuscript, the web-based version of the model is deployed on a
cloud Shiny hosting service (e.g., shinyapps.io). The same R/Shiny code
contained in this repository can be used for either:

- local deployment (as described above), or
- cloud deployment on a Shiny hosting platform.

Cloud deployment uses the same Shiny application code but a different
hosting infrastructure, as discussed in the Limitations and Shiny
hosting sections of the manuscript.

---

## 2. Environment information and dependencies

### 2.1 Software versions

The following software versions were used in the original analysis:

- Operating system: Windows 10 / 11 (64-bit)
- R: 4.5.1
- Python: 3.9.13

### 2.2 R dependencies (for the Shiny app)

The Shiny application depends on the following R packages:

- `shiny`
- `dplyr`
- `ggplot2`
- `DT`
- `readr`

These packages can be installed using the helper script
`install_packages.R`:

    source("install_packages.R")

### 2.3 Python dependencies and environment configuration

The Python scripts in the `code/` directory (e.g.
`01_logistic_model_development.py`, `02_model_validation.py`,
`03_generate_figures.py`) were developed and tested with the following
Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `statsmodels`
- `matplotlib`
- `joblib`

A minimal `requirements.txt` file is provided to allow users to recreate
the Python environment:

    pip install -r requirements.txt

Alternatively, a Conda environment can be created using `environment.yml`:

    conda env create -f environment.yml
    conda activate anvugib-rebleeding

Both `requirements.txt` and `environment.yml` are included in this
repository and specify the package versions used in the original analysis.

---

## 3. Repository structure (example)

    .
    ├─ code/
    │   ├─ 01_logistic_model_development.py
    │   ├─ 02_model_validation.py
    │   └─ 03_generate_figures.py
    ├─ shiny_app/
    │   ├─ app.R
    │   ├─ global.R      # if applicable
    │   ├─ ui.R          # if applicable
    │   └─ server.R      # if applicable
    ├─ data/
    │   └─ anvugib_final_analysis.csv   # final analysis dataset (not necessarily public)
    ├─ results/      # generated locally by the Python scripts
    ├─ figures/      # generated locally by the Python scripts
    ├─ models/       # generated locally by the Python scripts
    ├─ install_packages.R
    ├─ requirements.txt
    ├─ environment.yml
    └─ README.md

The `data/`, `results/`, `figures/`, and `models` folders are intended
for local use and do not need to be shared publicly if they contain
sensitive or intermediate data.
