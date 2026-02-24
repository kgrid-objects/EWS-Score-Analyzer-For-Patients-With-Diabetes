# EWS Score Analyzer

### Early Warning Score Threshold Evaluation & Diagnostic Performance Tool

A lightweight Python and JavaScript toolkit for analyzing **Early
Warning Scores (EWS)** in hospitalized patients with diabetes.

This project evaluates the predictive performance of EWS thresholds by
extracting each patient's maximum risk score and calculating key
diagnostic metrics including **sensitivity, specificity, positive
predictive value (PPV), negative predictive value (NPV), and accuracy
with 95% confidence intervals**.

Designed for **clinical researchers, health data scientists, quality
improvement teams, and digital health developers**.

------------------------------------------------------------------------

## Why This Tool Exists

When validating an Early Warning Score model, researchers often need to:

-   Identify each patient's highest recorded risk during hospitalization
-   Compare score distributions by outcome
-   Evaluate threshold performance
-   Quantify diagnostic accuracy with confidence intervals
-   Perform subgroup analyses (e.g., demographic or treatment groups)

This repository standardizes that workflow for reproducible EWS
validation and threshold optimization.

------------------------------------------------------------------------

## Keywords

Early Warning Score, EWS validation, clinical risk scoring, diagnostic
accuracy, sensitivity and specificity, predictive performance
evaluation, threshold analysis, clinical informatics, hospital
analytics, diabetes outcomes research, digital health tools, health data
science

------------------------------------------------------------------------

## Core Features

-   Extracts **maximum EWS per patient**
-   When given survival outcomes, it computes:
    -   Sensitivity
    -   Specificity
    -   PPV
    -   NPV
    -   Accuracy
    -   95% confidence intervals
-   Supports **custom maximum EWS threshold selection**
-   Optional **subgroup analysis**
-   Distribution visualization by outcome
-   Works in:
    -   Python (research workflows)
    -   JavaScript (web applications / dashboards)
    -   Standalone browser interface (no backend required)

------------------------------------------------------------------------

## Repository Structure

    ews_analysis.py        # Python implementation
    ews_analysis.js        # JavaScript implementation
    ews_analysis.html      # Ready-to-use web app implementation
    index.html             # Information Page describing the repo as Knowledge Object
    ews_sample_data.csv    # Example dataset
    docs/                  # Supporting documentation

------------------------------------------------------------------------

## Python Usage

``` python
import pandas as pd
from ews_analysis import analyze_ews

df = pd.read_csv("ews_sample_data.csv")
score_cols = [c for c in df.columns if c.startswith("ews_")]

results = analyze_ews(
    df,
    score_cols=score_cols,
    outcome_col="outcome",
    threshold=60,
    group_col="group"  # optional
)

print(results)
```

------------------------------------------------------------------------

## JavaScript Usage

``` html
<script src="ews_analysis.js"></script>
<script>
const results = EWSAnalysis.analyzeEWS({
  data: rows,
  scoreCols: scoreCols,
  outcomeCol: "outcome",
  threshold: 60
});

console.log(results);
</script>
```

------------------------------------------------------------------------

## Example Applications

-   Clinical research validation studies\
-   Early Warning Score threshold optimization\
-   Quality improvement analytics in hospital systems\
-   Risk stratification model evaluation\
-   Digital health dashboard integration

------------------------------------------------------------------------

## License

MIT License


