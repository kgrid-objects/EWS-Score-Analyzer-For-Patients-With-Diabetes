"""
Early Warning System (EWS) Score Analyzer
==========================================
Processes patient-level EWS scores to:
  1. Compute the highest EWS score per patient
  2. Plot the distribution of highest scores
  3. Compute diagnostic accuracy metrics at a user-defined cutoff threshold

Input DataFrame format:
  - One row per patient
  - Multiple columns of EWS scores (e.g., score_1, score_2, ..., score_N)
  - One binary outcome column:
        1 = survived and discharged
        0 = died in hospital

Usage:
    results = analyze_ews(df, score_cols, outcome_col, threshold=60)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


# ---------------------------------------------------------------------------
# Core Analysis Function
# ---------------------------------------------------------------------------

def analyze_ews(
    df: pd.DataFrame,
    score_cols: list[str],
    outcome_col: str,
    threshold: float = 60,
    plot: bool = True,
    group_col: str = None,
) -> dict:
    """
    Analyze EWS scores and compute diagnostic accuracy metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Patient-level data. One row per patient.
    score_cols : list[str]
        Column names containing EWS scores (one column per time point).
        Alternatively, pass a single column name if highest score is pre-computed.
    outcome_col : str
        Binary outcome column. 1 = survived/discharged, 0 = died in hospital.
    threshold : float
        EWS cutoff. Patients with highest score >= threshold are flagged as
        high-risk (predicted to die). Default = 60.
    plot : bool
        Whether to generate and display distribution plots. Default = True.
    group_col : str, optional
        Column name for subgroup analysis (e.g., 'treatment_group').
        If provided, metrics are computed per group in addition to overall.

    Returns
    -------
    dict with keys:
        'highest_scores'  : pd.Series of highest EWS score per patient
        'metrics'         : dict of overall diagnostic accuracy metrics
        'group_metrics'   : dict of per-group metrics (if group_col provided)
        'confusion_matrix': 2x2 array [[TN, FP], [FN, TP]]
    """

    # --- 1. Compute highest score per patient --------------------------------
    if isinstance(score_cols, str):
        score_cols = [score_cols]

    df = df.copy()
    df["_highest_score"] = df[score_cols].max(axis=1)
    df["_outcome"] = df[outcome_col]

    highest_scores = df["_highest_score"]

    # --- 2. Apply threshold to get predictions --------------------------------
    # Prediction: 1 = predicted death (score >= threshold), 0 = predicted survival
    df["_predicted_death"] = (df["_highest_score"] >= threshold).astype(int)

    # Actual death: outcome == 0
    df["_actual_death"] = (df["_outcome"] == 0).astype(int)

    # --- 3. Compute overall metrics -------------------------------------------
    overall_metrics = _compute_metrics(df["_actual_death"], df["_predicted_death"], threshold)

    # --- 4. Per-group metrics (optional) --------------------------------------
    group_metrics = {}
    if group_col is not None:
        for grp, grp_df in df.groupby(group_col):
            group_metrics[grp] = _compute_metrics(
                grp_df["_actual_death"], grp_df["_predicted_death"], threshold
            )

    # --- 5. Confusion matrix --------------------------------------------------
    tn = int(((df["_actual_death"] == 0) & (df["_predicted_death"] == 0)).sum())
    fp = int(((df["_actual_death"] == 0) & (df["_predicted_death"] == 1)).sum())
    fn = int(((df["_actual_death"] == 1) & (df["_predicted_death"] == 0)).sum())
    tp = int(((df["_actual_death"] == 1) & (df["_predicted_death"] == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])

    # --- 6. Print summary -----------------------------------------------------
    _print_summary(overall_metrics, group_metrics, cm, threshold)

    # --- 7. Plot --------------------------------------------------------------
    if plot:
        _plot_distribution(df, highest_scores, threshold, group_col)

    return {
        "highest_scores": highest_scores,
        "metrics": overall_metrics,
        "group_metrics": group_metrics,
        "confusion_matrix": cm,
        "data": df,
    }


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------

def _compute_metrics(actual_death: pd.Series, predicted_death: pd.Series, threshold: float) -> dict:
    """Compute diagnostic accuracy metrics and 95% CIs."""

    n = len(actual_death)
    tp = int(((actual_death == 1) & (predicted_death == 1)).sum())
    tn = int(((actual_death == 0) & (predicted_death == 0)).sum())
    fp = int(((actual_death == 0) & (predicted_death == 1)).sum())
    fn = int(((actual_death == 1) & (predicted_death == 0)).sum())

    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv          = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    accuracy     = (tp + tn) / n if n > 0 else np.nan
    prevalence   = (tp + fn) / n if n > 0 else np.nan

    # 95% Wilson confidence intervals for accuracy
    acc_ci = _wilson_ci(tp + tn, n)

    return {
        "n": n,
        "threshold": threshold,
        "prevalence": prevalence,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "accuracy": accuracy,
        "accuracy_95ci": acc_ci,
    }


def _wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0, center - margin), min(1, center + margin))


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_summary(metrics: dict, group_metrics: dict, cm: np.ndarray, threshold: float):
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  EWS DIAGNOSTIC ACCURACY  |  Cutoff Threshold: {threshold}")
    print(sep)
    _print_metrics_block("OVERALL", metrics)

    if group_metrics:
        for grp, gm in group_metrics.items():
            _print_metrics_block(f"GROUP: {grp}", gm)

    print(f"\n  Confusion Matrix (rows=Actual, cols=Predicted)")
    print(f"  {'':20s}  Pred Survived  Pred Died")
    print(f"  {'Actual Survived':20s}  {cm[0,0]:>13,}  {cm[0,1]:>9,}")
    print(f"  {'Actual Died':20s}  {cm[1,0]:>13,}  {cm[1,1]:>9,}")
    print(sep + "\n")


def _print_metrics_block(label: str, m: dict):
    ci = m["accuracy_95ci"]
    print(f"\n  [{label}]")
    print(f"  Patients        : {m['n']:,}")
    print(f"  Prevalence      : {m['prevalence']*100:.2f}%")
    print(f"  Accuracy        : {m['accuracy']*100:.2f}%  (95% CI: {ci[0]*100:.2f}% – {ci[1]*100:.2f}%)")
    print(f"  Sensitivity     : {m['sensitivity']*100:.2f}%")
    print(f"  Specificity     : {m['specificity']*100:.2f}%")
    print(f"  PPV             : {m['PPV']*100:.2f}%")
    print(f"  NPV             : {m['NPV']*100:.2f}%")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_distribution(df: pd.DataFrame, highest_scores: pd.Series, threshold: float, group_col: str):
    """Plot histogram of highest EWS scores, split by outcome and optionally by group."""

    survived = df.loc[df["_actual_death"] == 0, "_highest_score"]
    died     = df.loc[df["_actual_death"] == 1, "_highest_score"]

    n_groups = df[group_col].nunique() if group_col else 1
    n_cols   = 1 + n_groups if group_col else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), squeeze=False)

    # Overall plot
    _draw_histogram(axes[0, 0], survived, died, threshold, "Overall")

    # Per-group plots
    if group_col:
        for i, (grp, grp_df) in enumerate(df.groupby(group_col)):
            g_survived = grp_df.loc[grp_df["_actual_death"] == 0, "_highest_score"]
            g_died     = grp_df.loc[grp_df["_actual_death"] == 1, "_highest_score"]
            _draw_histogram(axes[0, i + 1], g_survived, g_died, threshold, f"Group: {grp}")

    plt.suptitle("Distribution of Highest EWS Score per Patient", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("ews_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Plot saved to: ews_distribution.png")


def _draw_histogram(ax, survived: pd.Series, died: pd.Series, threshold: float, title: str):
    bins = np.linspace(0, 100, 41)  # bins of width 2.5 across 0–100

    ax.hist(survived, bins=bins, alpha=0.6, color="steelblue", label=f"Survived (n={len(survived):,})")
    ax.hist(died,     bins=bins, alpha=0.7, color="firebrick", label=f"Died (n={len(died):,})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")

    ax.set_xlabel("Highest EWS Score", fontsize=11)
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 100)


# ---------------------------------------------------------------------------
# Demo / Example Usage
# ---------------------------------------------------------------------------

#if __name__ == "__main__":
#
    # --- Generate synthetic data resembling the abstract's cohort ------------
#    np.random.seed(42)
#    n_medical  = 500
#    n_surgical = 300

#    def simulate_patients(n, death_rate, high_score_if_died_mean, high_score_if_survived_mean, group_label):
#        n_died     = int(n * death_rate)
#        n_survived = n - n_died

        # Simulate 10 EWS score readings per patient (every 15 min over ~2.5 hrs sample)
        # In practice these would be all readings during the full hospitalization
#        def make_scores(n_pts, peak_mean, peak_std=15):
#            peaks = np.clip(np.random.normal(peak_mean, peak_std, n_pts), 0, 100)
#            scores = []
#            for peak in peaks:
                # Other readings are lower than the peak
#                readings = np.clip(np.random.normal(peak * 0.7, 10, 9), 0, 100)
#                readings = np.append(readings, peak)
#                scores.append(readings)
#            return np.array(scores)

#        scores_died     = make_scores(max(n_died, 1),     high_score_if_died_mean)
#        scores_survived = make_scores(n_survived, high_score_if_survived_mean)

#        score_cols = [f"ews_{i}" for i in range(10)]
#        df_died     = pd.DataFrame(scores_died[:n_died] if n_died > 0 else np.empty((0, 10)), columns=score_cols)
#        df_survived = pd.DataFrame(scores_survived, columns=score_cols)

#        df_died["outcome"]     = 0  # died
#        df_survived["outcome"] = 1  # survived

#        df = pd.concat([df_died, df_survived], ignore_index=True)
#        df["group"] = group_label
#        return df, score_cols

#    df_med, score_cols = simulate_patients(n_medical,  death_rate=0.0086, high_score_if_died_mean=75, high_score_if_survived_mean=35, group_label="Medical")
#    df_surg, _        = simulate_patients(n_surgical,  death_rate=0.001,  high_score_if_died_mean=65, high_score_if_survived_mean=38, group_label="Surgical")

#   df_all = pd.concat([df_med, df_surg], ignore_index=True)

#   print("Sample data (first 5 rows):")
#   print(df_all[score_cols[:3] + ["outcome", "group"]].head())

    # --- Run analysis ---------------------------------------------------------
#   results = analyze_ews(
#       df=df_all,
#       score_cols=score_cols,
#       outcome_col="outcome",
#       threshold=60,
#       plot=True,
#       group_col="group",
#   )

#---------------
# Read input data from sample data file
#---------------

import pandas as pd
from ews_analysis import analyze_ews

df = pd.read_csv("ews_sample_data.csv")
score_cols = [c for c in df.columns if c.startswith("ews_")]

results = analyze_ews(df, score_cols=score_cols, outcome_col="outcome", threshold=60, group_col="group")
