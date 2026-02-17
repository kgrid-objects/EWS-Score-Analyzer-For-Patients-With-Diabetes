# EWS-Score-Analyzer-For-Patients-With-Diabetes
This algorithm processes patient-level EWS data to extract each patient's highest risk score during hospitalization and plots its distribution by outcome. Given a user-defined threshold, it computes sensitivity, specificity, PPV, NPV, and accuracy with 95% confidence intervals, with optional subgroup analysis.

#### Using the JS Module

TO USE THE JS MODULE, DO THIS:

import pandas as pd
from ews_analysis import analyze_ews

df = pd.read_csv("ews_sample_data.csv")
score_cols = [c for c in df.columns if c.startswith("ews_")]

results = analyze_ews(df, score_cols=score_cols, outcome_col="outcome", threshold=60, group_col="group")

TO USE THE JS MODULE IN AN HTML FILE, DO THIS:

<script src="ews_analysis.js"></script>
<script>
  const results = EWSAnalysis.analyzeEWS({ data: rows, scoreCols, outcomeCol: 'outcome', threshold: 60 });
</script>


