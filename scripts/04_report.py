"""
Generate the comparison figure and summary table from
outputs/model_comparison_results.csv.

Usage:
    python scripts/04_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import pandas as pd
import matplotlib.pyplot as plt


def main():
    if not config.RESULTS_CSV.exists():
        print("ERROR: run 03_ml_pipeline.py first.")
        sys.exit(1)

    df = pd.read_csv(config.RESULTS_CSV)
    best = df.loc[df.groupby("feature_block")["mean_auc"].idxmax()].sort_values(
        "mean_auc", ascending=False
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"clinical": "#4C72B0", "radiomics": "#DD8452", "combined": "#55A868"}
    bars = ax.bar(best["feature_block"], best["mean_auc"],
                   yerr=best["std_auc"], capsize=5,
                   color=[colors.get(b, "#888") for b in best["feature_block"]])

    for bar, (_, row) in zip(bars, best.iterrows()):
        stars = "***" if row.permutation_p < 0.001 else "**" if row.permutation_p < 0.01 \
            else "*" if row.permutation_p < 0.05 else "ns"
        ax.text(bar.get_x() + bar.get_width() / 2, row.mean_auc + row.std_auc + 0.02,
                f"{row.model}\nAUC={row.mean_auc:.2f}\n{stars}",
                ha="center", va="bottom", fontsize=9)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC (nested CV)")
    ax.set_title("Lung metastasis prediction: clinical vs radiomics vs combined\n"
                  "(Soft-Tissue-Sarcoma, n=51, best model per block)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig_path = config.OUTPUT_DIR / "block_comparison.png"
    plt.savefig(fig_path, dpi=200)
    print(f"Saved figure to {fig_path}")

    print("\n=== Best model per feature block ===")
    print(best[["feature_block", "model", "mean_auc", "std_auc", "permutation_p", "n_features"]]
          .to_string(index=False))

    summary_path = config.OUTPUT_DIR / "summary.md"
    with open(summary_path, "w") as f:
        f.write("# Soft-Tissue-Sarcoma Lung Metastasis Prediction — Block Comparison\n\n")
        f.write(best[["feature_block", "model", "mean_auc", "std_auc", "permutation_p", "n_features", "n_patients"]]
                .to_markdown(index=False))
        f.write("\n\nCohort: n=51 (TCIA/IDC Soft-Tissue-Sarcoma, Vallieres et al. 2015). "
                "Small-cohort caveat applies — treat AUCs as directional.\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
