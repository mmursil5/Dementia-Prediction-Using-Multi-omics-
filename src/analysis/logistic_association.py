"""
Proteins-only association analysis (CSV version, polished)

- Adjusted logistic regression per protein (incident dementia: 0/1)
  -> OR per 1-SD, 95% CI, p, BH–FDR
- Reads outcome + covariates once; then reads ONE protein column at a time from CSV.
- Outputs tidy results + a colorful FDR-annotated forest plot.

Requires: pandas, numpy, statsmodels, scipy, matplotlib
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =========================
# CONFIG
# =========================
CSV_PATH = "multiomics_features.csv"   # your CSV dataset
OUTCOME = "diagn"                                   # 0/1 incident dementia

# Covariates for adjustment (edit if needed)
COVARS = ["Sex","Age","Education","Ethnicity","APOE4"]

# List proteins (or set to None to auto-detect)
PROTEIN_COLUMNS = ["GFAP", "APOE", "NEFL", "GDF15", "ACTA2", "EDA2R", "LECT2", "LTBP2", "SCARF2", "ELN", "WNT9A"]

# Transformations
LOG1P_FEATURES = True
ZSCORE_FEATURES = True
ROBUST_SE = True

# Thresholds / outputs
MAX_NA_FRAC = 0.20
MIN_SAMPLE = 200
OUTPUT_PREFIX = "./proteins_assoc"
MAKE_FOREST_PLOT = True
TOP_N_PLOT = 25
# =========================

# ---------- Load outcome + covariates once ----------
usecols = [OUTCOME] + COVARS
df_cov = pd.read_csv(CSV_PATH, usecols=usecols)
df_cov = df_cov.replace([np.inf, -np.inf], np.nan)
df_cov = df_cov.dropna(subset=[OUTCOME] + COVARS)

y_all = df_cov[OUTCOME].astype(int)
Xcov = pd.get_dummies(df_cov[COVARS], drop_first=True)
Xcov = sm.add_constant(Xcov, has_constant="add")

n_base = len(df_cov)

# ---------- Auto-detect protein columns if None ----------
if PROTEIN_COLUMNS is None:
    hdr = pd.read_csv(CSV_PATH, nrows=0)
    exclude = set([OUTCOME] + COVARS)
    PROTEIN_COLUMNS = [c for c in hdr.columns if c not in exclude]

# Deduplicate
PROTEIN_COLUMNS = list(dict.fromkeys(PROTEIN_COLUMNS))
print(f"Analyzing {len(PROTEIN_COLUMNS)} proteins; base N = {n_base}")

results = []

# ---------- Fit logistic regression per protein ----------
def fit_logistic_for_protein(protein_col: str):
    try:
        ser = pd.read_csv(CSV_PATH, usecols=[protein_col])[protein_col]
    except Exception as e:
        return {"feature": protein_col, "n": 0, "status": f"missing_column: {e}"}

    df = pd.concat([y_all.reset_index(drop=True),
                    Xcov.reset_index(drop=True),
                    ser.reset_index(drop=True)], axis=1).rename(columns={protein_col: "protein"})

    n_nonmissing = df["protein"].notna().sum()
    na_frac = 1.0 - (n_nonmissing / n_base)
    if na_frac > MAX_NA_FRAC:
        return {"feature": protein_col, "n": int(n_nonmissing), "status": f"too_many_missing({na_frac:.2f})"}

    df = df.dropna(subset=["protein"])
    n_total = len(df)
    if n_total < MIN_SAMPLE:
        return {"feature": protein_col, "n": n_total, "status": "too_few_samples"}

    # Transform
    x = df["protein"].astype(float).copy()
    if LOG1P_FEATURES and (x.dropna() >= 0).all():
        x = np.log1p(x)
    if ZSCORE_FEATURES:
        mu, sd = x.mean(skipna=True), x.std(skipna=True)
        if (sd is None) or (not np.isfinite(sd)) or (sd <= 0):
            return {"feature": protein_col, "n": n_total, "status": "zero_variance"}
        x = (x - mu) / sd

    X = pd.concat([df[Xcov.columns], x.rename("protein")], axis=1).astype(float)
    y = df[OUTCOME].astype(int).values

    try:
        fit = sm.Logit(y, X.values).fit(disp=False)
        if ROBUST_SE:
            fit_rb = sm.Logit(y, X.values).fit(disp=False, cov_type="HC3")
            coef, se = float(fit.params[-1]), float(fit_rb.bse[-1])
        else:
            coef, se = float(fit.params[-1]), float(fit.bse[-1])

        OR = float(np.exp(coef))
        z_crit = 1.959963984540054
        CI_low = float(np.exp(coef - z_crit * se))
        CI_high = float(np.exp(coef + z_crit * se))
        pval = float(2 * (1 - norm.cdf(abs(coef / se))))

        return {"feature": protein_col, "n": n_total, "OR_per_SD": OR,
                "CI95_low": CI_low, "CI95_high": CI_high, "p": pval, "status": "ok"}
    except Exception as e:
        return {"feature": protein_col, "n": n_total, "status": f"fit_fail: {e}"}

# ---------- Run ----------
for i, prot in enumerate(PROTEIN_COLUMNS, 1):
    out = fit_logistic_for_protein(prot)
    if out: results.append(out)
    if i % 50 == 0:
        print(f"... processed {i}/{len(PROTEIN_COLUMNS)} proteins")

res = pd.DataFrame(results)
ok = res[res["status"] == "ok"].copy()
if len(ok):
    ok = ok.sort_values("p")
    ok["FDR"] = multipletests(ok["p"], method="fdr_bh")[1]
    ok = ok.reset_index(drop=True)

# ---------- Save ----------
ok.to_csv(f"{OUTPUT_PREFIX}_logistic_associations.csv", index=False)
res.to_csv(f"{OUTPUT_PREFIX}_logistic_all_status.csv", index=False)
print(f"Saved: {OUTPUT_PREFIX}_logistic_associations.csv (n={len(ok)})")

# ---------- Forest Plot (clean labels, no FDR on y-axis) ----------
if MAKE_FOREST_PLOT and len(ok):
    top = ok.sort_values(["FDR", "p"]).head(TOP_N_PLOT).copy()
    ypos = np.arange(len(top))[::-1]

    # Only metabolite names in y-axis labels
    top["label"] = top["feature"]

    colors = cm.tab10(np.linspace(0, 1, len(top)))

    plt.figure(figsize=(8, 0.5*len(top) + 2))
    for y_i, lo, hi, c in zip(ypos, top["CI95_low"], top["CI95_high"], colors):
        plt.plot([lo, hi], [y_i, y_i], linewidth=6, color=c)

    plt.axvline(1.0, linestyle="--", linewidth=2.5, color="red")

    sig_mask = top["FDR"] < 0.05
    plt.scatter(top["OR_per_SD"][sig_mask], ypos[sig_mask], s=300,
                color=colors[sig_mask], edgecolors="black", zorder=3)
    plt.scatter(top["OR_per_SD"][~sig_mask], ypos[~sig_mask], s=300,
                facecolors="none", edgecolors=colors[~sig_mask], linewidth=3.5, zorder=3)

    plt.yticks(ypos, top["label"].tolist(), fontsize=20)
    plt.xticks(fontsize=18)
    plt.xlabel("Odds Ratio per 1-SD (95% CI)", fontsize=20)
    plt.title("Incident AD", fontsize=20, weight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_AD.pdf", bbox_inches="tight", dpi=400)
    plt.show()

