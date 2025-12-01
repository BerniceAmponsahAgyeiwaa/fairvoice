import os
import pandas as pd
import matplotlib.pyplot as plt


########################################
# Utility: Compute accuracy by group
########################################

def compute_accuracy_by_group(df, group_col):
    return (
        df.groupby(group_col, observed=False)
          .apply(lambda g: (g["true_label"] == g["pred_label"]).mean(), include_groups=False)
    )


########################################
# Main tradeoff analysis
########################################

def tradeoff_analysis(eval_path, out_dir):
    """
    eval_path: CSV with columns:
        file, ActorID, Age, Sex, Race, Ethnicity, true_label, pred_label
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(eval_path)

    # -------------------------------
    # Compute accuracy per group
    # -------------------------------

    acc_race = compute_accuracy_by_group(df, "Race")
    acc_sex = compute_accuracy_by_group(df, "Sex")
    acc_age = compute_accuracy_by_group(df, "Age")

    overall_acc = (df["true_label"] == df["pred_label"]).mean()

    # -------------------------------
    # Compute fairness gaps
    # -------------------------------

    gap_race = acc_race.max() - acc_race.min()
    gap_sex  = acc_sex.max()  - acc_sex.min()
    gap_age  = acc_age.max()  - acc_age.min()

    # Save summary table
    summary = pd.DataFrame({
        "Group Type": ["Race", "Sex", "Age"],
        "Fairness Gap": [gap_race, gap_sex, gap_age],
        "Min Accuracy": [acc_race.min(), acc_sex.min(), acc_age.min()],
        "Max Accuracy": [acc_race.max(), acc_sex.max(), acc_age.max()],
    })
    summary.to_csv(os.path.join(out_dir, "fairness_gaps.csv"), index=False)


    ########################################
    # Plot Tradeoff Curve
    #
    # - X-axis: fairness gap
    # - Y-axis: accuracy
    ########################################

    plt.figure(figsize=(7, 5))
    gaps = [gap_race, gap_sex, gap_age]
    labels = ["Race", "Sex", "Age"]

    plt.scatter(gaps, [overall_acc]*3, s=150)

    for g, label in zip(gaps, labels):
        plt.text(g + 0.002, overall_acc, label, fontsize=12)

    plt.xlabel("Fairness Gap (max acc - min acc)")
    plt.ylabel("Overall Accuracy")
    plt.title("Accuracy–Fairness Tradeoff")
    plt.grid(True)

    out_plot = os.path.join(out_dir, "tradeoff_curve.png")
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n=== Tradeoff Analysis Complete ===")
    print(f"Overall accuracy: {overall_acc:.3f}")
    print(f"Fairness gaps saved to: {out_dir}")
    print(f"Tradeoff curve saved to: {out_plot}")


########################################
# Run standalone
########################################

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    eval_path = os.path.join(project_root, "evaluation_results", "bias_report.csv")
    out_dir   = os.path.join(project_root, "evaluation_results", "tradeoff_analysis")

    tradeoff_analysis(eval_path, out_dir)
