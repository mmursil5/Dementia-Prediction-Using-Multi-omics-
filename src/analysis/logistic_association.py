"""
Multivariable-adjusted logistic regression (per protein/metabolite)
with robust HC3 SE and BH–FDR.

- Outcome: binary (e.g., incident AD)
- Covariates: specified in COVARS below.
"""

import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def fit_logistic_for_feature(
    csv_path: str,
    outcome: str,
    covars,
    feature_col: str,
    max_na_frac: float = 0.20,
    min_sample: int = 200,
    log1p_features: bool = True,
    zscore_features: bool = True,
    robust_se: bool = True,
):
    usecols = [outcome] + covars
    df_cov = pd.read_csv(csv_path, usecols=usecols)
    df_cov = df_cov.replace([np.inf, -np.inf], np.nan)
    df_cov = df_cov.dropna(subset=[outcome] + covars)

    y_all = df_cov[outcome].astype(int)
    Xcov = pd.get_dummies(df_cov[covars], drop_first=True)
    Xcov = sm.add_constant(Xcov, has_constant="add")

    n_base = len(df_cov)

    try:
        ser = pd.read_csv(csv_path, usecols=[feature_col])[feature_col]
    except Exception as e:
        return {"feature": feature_col, "n": 0, "status": f"missing_column: {e}"}

    df = pd.concat(
        [
            y_all.reset_index(drop=True),
            Xcov.reset_index(drop=True),
            ser.reset_index(drop=True),
        ],
        axis=1,
    ).rename(columns={feature_col: "feature"})

    n_nonmissing = df["feature"].notna().sum()
    na_frac = 1.0 - (n_nonmissing / n_base)
    if na_frac > max_na_frac:
        return {
            "feature": feature_col,
            "n": int(n_nonmissing),
            "status": f"too_many_missing({na_frac:.2f})",
        }

    df = df.dropna(subset=["feature"])
    n_total = len(df)
    if n_total < min_sample:
        return {"feature": feature_col, "n": n_total, "status": "too_few_samples"}

    x = df["feature"].astype(float).copy()
    if log1p_features and (x.dropna() >= 0).all():
        x = np.log1p(x)
    if zscore_features:
        mu, sd = x.mean(skipna=True), x.std(skipna=True)
        if (sd is None) or (not np.isfinite(sd)) or (sd <= 0):
            return {"feature": feature_col, "n": n_total, "status": "zero_variance"}
        x = (x - mu) / sd

    X = pd.concat([df[Xcov.columns], x.rename("feature")], axis=1).astype(float)
    y = df[outcome].astype(int).values

    try:
        fit = sm.Logit(y, X.values).fit(disp=False)
        if robust_se:
            fit_rb = sm.Logit(y, X.values).fit(disp=False, cov_type="HC3")
            coef, se = float(fit.params[-1]), float(fit_rb.bse[-1])
        else:
            coef, se = float(fit.params[-1]), float(fit.bse[-1])

        OR = float(np.exp(coef))
        z_crit = 1.959963984540054
        CI_low = float(np.exp(coef - z_crit * se))
        CI_high = float(np.exp(coef + z_crit * se))
        pval = float(2 * (1 - norm.cdf(abs(coef / se))))

        return {
            "feature": feature_col,
            "n": n_total,
            "OR_per_SD": OR,
            "CI95_low": CI_low,
            "CI95_high": CI_high,
            "p": pval,
            "status": "ok",
        }
    except Exception as e:
        return {"feature": feature_col, "n": n_total, "status": f"fit_fail: {e}"}


def run_association(
    csv_path: str,
    outcome: str,
    covars,
    feature_columns,
    output_prefix: str,
    make_forest_plot: bool = True,
    top_n_plot: int = 25,
):
    results = []
    for i, feat in enumerate(feature_columns, 1):
        res = fit_logistic_for_feature(
            csv_path=csv_path,
            outcome=outcome,
            covars=covars,
            feature_col=feat,
        )
        results.append(res)
        if i % 50 == 0:
            print(f"... processed {i}/{len(feature_columns)} features")

    res_df = pd.DataFrame(results)
    ok = res_df[res_df["status"] == "ok"].copy()
    if len(ok):
        ok = ok.sort_values("p")
        ok["FDR"] = multipletests(ok["p"], method="fdr_bh")[1]
        ok = ok.reset_index(drop=True)

    ok.to_csv(f"{output_prefix}_logistic_associations.csv", index=False)
    res_df.to_csv(f"{output_prefix}_logistic_all_status.csv", index=False)
    print(f"Saved: {output_prefix}_logistic_associations.csv (n={len(ok)})")

    if make_forest_plot and len(ok):
        top = ok.sort_values(["FDR", "p"]).head(top_n_plot).copy()
        ypos = np.arange(len(top))[::-1]

        top["label"] = top["feature"]
        colors = cm.tab10(np.linspace(0, 1, len(top)))

        plt.figure(figsize=(8, 0.5 * len(top) + 2))
        for y_i, lo, hi, c in zip(ypos, top["CI95_low"], top["CI95_high"], colors):
            plt.plot([lo, hi], [y_i, y_i], linewidth=6, color=c)

        plt.axvline(1.0, linestyle="--", linewidth=2.5, color="red")

        sig_mask = top["FDR"] < 0.05
        plt.scatter(
            top["OR_per_SD"][sig_mask],
            ypos[sig_mask],
            s=300,
            color=colors[sig_mask],
            edgecolors="black",
            zorder=3,
        )
        plt.scatter(
            top["OR_per_SD"][~sig_mask],
            ypos[~sig_mask],
            s=300,
            facecolors="none",
            edgecolors=colors[~sig_mask],
            linewidth=3.5,
            zorder=3,
        )

        plt.yticks(ypos, top["label"].tolist(), fontsize=12)
        plt.xlabel("Odds Ratio per 1-SD (95% CI)", fontsize=14)
        plt.title("Incident dementia", fontsize=14, weight="bold")
        plt.tight.tight_layout()
        plt.savefig(f"{output_prefix}_forest.pdf", bbox_inches="tight", dpi=300)
        plt.show()


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--outcome", type=str, default="diagn")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="./omics_assoc",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Optional list of features; if omitted, auto-detect all non-covariates.",
    )
    args = parser.parse_args()

    # Adjust to match your exact covariates
    COVARS = [
        "Sex",
        "Education years",
        "Smoking",
        "BMI",
        "Age",
        "IPAQ",
        "Sleep",
        "Ethnicity",
        "APOE4",
    ]

    if args.features is None:
        hdr = pd.read_csv(args.csv, nrows=0)
        exclude = set([args.outcome] + COVARS)
        feature_columns = [c for c in hdr.columns if c not in exclude]
    else:
        feature_columns = args.features

    run_association(
        csv_path=args.csv,
        outcome=args.outcome,
        covars=COVARS,
        feature_columns=feature_columns,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    cli()
