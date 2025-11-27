# Workflow for Data Preprocessing, Model Development, Validation and Deployment

This document provides a step-by-step description of the workflow used
to develop, validate, and deploy the 7-day rebleeding risk prediction
model for patients with acute non-variceal upper gastrointestinal
bleeding (ANVUGIB).

The description follows exactly the procedures reported in the
“Materials and Methods” section of the manuscript.

> **Data availability**
>
> In accordance with the Ethics Committee approval and institutional
> regulations, this repository does **not** contain any raw or
> patient-level data (including de-identified datasets). The scripts
> and documents provided here are intended to make the analysis steps
> transparent and reproducible for users who have appropriate access
> to the underlying hospital electronic medical record (EMR) data.

---

## 1. Study design, setting and data source

- Design: single-centre retrospective cohort study.
- Setting: General Hospital of Central Theater Command.
- Study periods:
  - January 2020 to December 2023 for model derivation  
  - January 2024 to August 2024 for temporal external validation
- Case identification:
  - All potentially eligible patients with ANVUGIB were screened
    through the hospital’s electronic medical record (EMR) system.
- Data collection:
  - Demographic characteristics, clinical symptoms, vital signs,
    laboratory parameters and in-hospital outcomes were extracted.
  - To ensure data quality, data extraction was performed independently
    by two investigators (double entry) and cross-checked.

All primary analyses were performed using FreeStatistics software
(V2.2.0), as stated in the manuscript. The scripts in this repository
mirror the same analytical steps for transparency but do not include
any EMR data.

---

## 2. Study population and cohort construction

### 2.1 Inclusion criteria

1. Age ≥ 18 years;  
2. Confirmed diagnosis of acute non-variceal upper gastrointestinal
   bleeding (ANVUGIB);  
3. Completion of upper endoscopy within 48 hours of admission with
   exclusion of variceal bleeding.

### 2.2 Exclusion criteria

1. Confirmed or suspected variceal bleeding due to esophageal/gastric
   varices;  
2. Concurrent upper gastrointestinal malignancy or history of upper
   gastrointestinal surgery;  
3. Self-discharge after admission or > 20% missing key variables;  
4. Pregnancy or lactation.

### 2.3 Derivation, internal validation and temporal validation cohorts

- From January 2020 to December 2023, **818** eligible patients with
  ANVUGIB were retrospectively included.
- These 818 patients were randomly assigned in a **7:3 ratio** to:
  - Training cohort: n = 572  
  - Internal validation cohort: n = 246  
- Randomisation was implemented programmatically using a 7:3 split, consistent with the manuscript.
- An additional **147** patients admitted between January 2024 and
  August 2024 were used as an **independent temporal validation cohort**.

The cohort construction (application of inclusion/exclusion criteria
and 7:3 split) is implemented programmatically but follows exactly the
rules above.

---

## 3. Outcome definition

### 3.1 Primary outcome: 7-day rebleeding

The primary outcome was **rebleeding within 7 days after admission**,
defined as any of the following after initial hemostasis
(pharmacological, endoscopic, or interventional):

1. Endoscopically confirmed fresh or active bleeding;  
2. A hemoglobin drop ≥ 2 g/dL after exclusion of other sources of
   blood loss;  
3. Recurrent hematemesis or melena with hemodynamic instability
   (systolic blood pressure < 90 mmHg or heart rate > 100/min);  
4. Requirement for repeat endoscopy, transcatheter arterial embolization,
   or surgery.

All rebleeding events were restricted to those occurring within
7 days after admission. A binary outcome variable (yes/no) was created
accordingly.

---

## 4. Predictors and data collection timeline

### 4.1 Candidate predictors collected

The following variables were collected as candidate predictors:

- **Demographics:** age, sex, smoking, alcohol use.  
- **Comorbidities:** hypertension, diabetes, coronary artery disease,
  hepatic dysfunction, chronic kidney disease.  
- **Medications:** anticoagulants, antiplatelets, NSAIDs.  
- **Clinical symptoms:** hematemesis, melena, syncope.  
- **Vital signs:** systolic blood pressure (SBP), pulse, bowel sounds.  
- **Laboratory tests:** hemoglobin, platelet count, albumin, creatinine,
  INR, prothrombin time (PT).  
- **Risk scores:** Glasgow–Blatchford score (GBS) and AIMS65.

### 4.2 Definitions of key covariates

Key covariates were defined exactly as in the manuscript, including:

- **Active bowel sounds:**  
  Assessed in a quiet setting with the patient supine. The stethoscope
  was placed in the right lower quadrant, then moved counterclockwise
  across four quadrants, with ≥ 1 minute of auscultation per quadrant.
  Active bowel sounds were defined as ≥ 10 sounds/min or the presence of
  high-pitched, hyperactive, metallic, or “tinkling” sounds. Initial
  assessment was performed by an attending or senior resident and
  verified by another physician. Discrepancies were adjudicated by a
  senior gastroenterologist. Patients with recent use of laxatives,
  prokinetics, or bowel stimulation within 6 hours were excluded from
  bowel sound analysis.

- **Altered mental status:**  
  Glasgow Coma Scale (GCS) ≤ 13 on admission.

- **Hepatic dysfunction:**  
  Serum ALT or AST > 2 × upper limit of normal (ULN) at admission.

- **Renal dysfunction:**  
  Serum creatinine > 110 μmol/L at admission.

### 4.3 Timing of data collection

- All candidate predictors were collected from admission to the time of
  the index endoscopy.
- Information related to the first endoscopic examination was obtained
  within 48 hours after admission.
- Rebleeding events within 7 days after admission were determined based
  on detailed review of the inpatient clinical course, including
  progress notes, repeat endoscopic examinations, and treatment records.

---

## 5. Handling of missing data, outliers and standardisation

### 5.1 Missing data

Missing data were handled exactly as follows:

- For candidate predictors with **< 5% missingness**:
  - **Single imputation** was performed:
    - Continuous variables: imputed using the median;
    - Categorical variables: imputed using the mode;
  - This approach was chosen given the minimal expected impact on effect
    estimates and model performance.

- For variables with **≥ 5% missingness**:
  - **Multiple imputation by chained equations (MICE)** with
    **10 imputations** was applied to better account for uncertainty.
  - Imputation was conducted under the assumption that data were missing
    at random.

### 5.2 Outliers

- Outliers in continuous variables were checked using boxplots.
- Values falling within clinically reasonable ranges were **retained**.
- Implausible values, if identified, were handled at the preprocessing
  stage (for example, set to missing and then handled via the
  imputation strategy above).

### 5.3 Standardisation of continuous variables

- Continuous variables were **standardized using Z-scores before
  modeling**.
- The standardisation step used the mean and standard deviation of the
  relevant cohort, consistent with the procedures implemented in
  FreeStatistics V2.2.0.

---

## 6. Model development

### 6.1 Model type

- The outcome was 7-day rebleeding (yes/no).
- A **multivariable logistic regression model** was applied to predict
  the probability of 7-day rebleeding.

### 6.2 Variable selection

- Variable selection was performed using **LASSO regression with
  10-fold cross-validation**.
- All prespecified candidate predictors (Section 4.1) were entered into
  the LASSO procedure.
- Predictors with non-zero coefficients at the optimal penalty parameter
  were considered candidate variables for the final model.

### 6.3 Final model fitting

- The variables selected by LASSO were then entered into a **multivariable
  logistic regression model** to develop the final prediction model for
  7-day rebleeding.
- The model was derived in the training cohort (70% of the 2020–2023
  data) and then applied, without modification, to the internal
  validation cohort and the temporal validation cohort.

The coefficients obtained from this final model are the ones used in the
prediction scripts and in the Shiny application.

---

## 7. Model validation

### 7.1 Internal validation

- The dataset from January 2020 to December 2023 was randomly split in a
  7:3 ratio into:
  - Training cohort (n = 572);
  - Internal validation cohort (n = 246).
- Model performance was assessed in the **internal validation cohort**
  using:
  - **Discrimination:**
    - ROC curve analysis with area under the curve (AUC) and 95%
      confidence intervals.
  - **Calibration:**
    - Calibration-in-the-large;
    - Calibration slope;
    - Hosmer–Lemeshow goodness-of-fit test;
    - Bootstrap-derived calibration plots.
  - **Clinical utility:**
    - Decision curve analysis (DCA) to quantify net benefit across a
      range of risk thresholds.

### 7.2 Temporal external validation

- Patients admitted between January and August 2024 (n = 147) served as
  an **independent temporal validation cohort**.
- The same evaluation methods were applied:
  - ROC–AUC and 95% confidence intervals;
  - Calibration metrics and calibration plots;
  - DCA for clinical utility.

### 7.3 Comparison with existing risk scores

- In the internal validation cohort, the AUCs of **AIMS65** and **GBS**
  were calculated for comparison with the new model.
- Differences in AUC were tested using **DeLong’s method**.
- DCA was further applied to compare the **net clinical benefit** of
  the new model, AIMS65 and GBS across different risk thresholds.

### 7.4 Stratified analysis by early intensive hemostatic intervention

- To evaluate the potential impact of different initial hemostatic
  strategies on model performance, a prespecified stratified analysis
  was conducted based on early intensive hemostatic intervention.
- The overall cohort was divided into:
  1. Patients who did **not** receive early intensive hemostatic
     intervention;  
  2. Patients who **did** receive early intensive hemostatic
     intervention.
- Early intensive hemostatic intervention was defined as the timely
  implementation, after the index bleeding event, of at least one active
  hemostatic modality, including early endoscopic hemostasis,
  transcatheter arterial embolization, or surgical intervention.

Importantly:

- The **regression coefficients of the primary model were kept
  unchanged**.
- Predicted probabilities of 7-day rebleeding were calculated for each
  patient based on the final model.
- ROC curves were developed and AUCs with 95% confidence intervals were
  estimated separately in each subgroup to assess discriminative
  performance under different initial treatment strategies.

---

## 8. Model performance evaluation and statistical software

- **Discrimination** was evaluated using ROC curves and AUCs with
  95% confidence intervals.
- **Calibration** was assessed using calibration-in-the-large, calibration
  slope, the Hosmer–Lemeshow test, and bootstrap-derived calibration
  plots.
- **Clinical utility** was evaluated using decision curve analysis.

All analyses were performed using **FreeStatistics software (V2.2.0)**.
Continuous variables were tested for normality using the Shapiro–Wilk
test. Normally distributed variables were reported as mean ± SD and
compared with the independent-samples t-test; non-normally distributed
variables were presented as median (IQR) and compared with the
Mann–Whitney U test. Categorical variables were expressed as counts (%)
and compared using the χ² test or Fisher’s exact test. A two-sided
P < 0.05 was considered statistically significant.

In this GitHub repository, Python scripts are provided to reproduce the
final multivariable logistic regression model and its validation on the
fully preprocessed analysis dataset, starting from
`data/anvugib_final_analysis.csv`, which is the export from
FreeStatistics.

---

## 9. Deployment of the prediction model (Shiny application)

- The final multivariable logistic regression model for 7-day rebleeding
  was implemented in an interactive **Shiny web application**.
- The app takes the selected predictors as inputs (e.g. syncope, pulse,
  bowel sounds, RDW, albumin) and returns the predicted probability of
  7-day rebleeding for an individual patient.
- The deployment process consists of:
  1. Embedding the final model coefficients and formula in the Shiny
     server code (`app.R`);
  2. Creating a user interface with input fields corresponding to the
     clinical predictors and a numerical output for the predicted risk;
  3. Optionally deploying the app to a Shiny cloud hosting service
     (e.g. shinyapps.io) for demonstration and research use.

The Shiny implementation is for **research and educational purposes
only**, consistent with the manuscript, and does not alter the model
development or validation procedures described above.
