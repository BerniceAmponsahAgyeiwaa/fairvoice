import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_predictions(csv_path):
    return pd.read_csv(csv_path)

def compute_accuracy_by_group(df, group_col):
    return (
        df.groupby(group_col, observed=False)
          .apply(lambda g: (g["true_label"] == g["pred_label"]).mean(), include_groups=False)
    )

def visualize_bias(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = load_predictions(csv_path)

    # --- Overall accuracy
    overall_acc = (df["true_label"] == df["pred_label"]).mean()

    # --- Accuracy by Race
    acc_by_race = compute_accuracy_by_group(df, "Race")
    acc_by_race.to_csv(os.path.join(output_dir, "accuracy_by_race.csv"))

    plt.figure(figsize=(8,5))
    sns.barplot(x=acc_by_race.index, y=acc_by_race.values)
    plt.title("Accuracy by Race")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_race.png"))
    plt.close()

    # --- Accuracy by Sex
    acc_by_sex = compute_accuracy_by_group(df, "Sex")
    acc_by_sex.to_csv(os.path.join(output_dir, "accuracy_by_sex.csv"))

    plt.figure(figsize=(6,5))
    sns.barplot(x=acc_by_sex.index, y=acc_by_sex.values)
    plt.title("Accuracy by Sex")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_sex.png"))
    plt.close()

    # --- Accuracy by Age (optional: bin age)
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 80], labels=["<30", "30-45", "45-60", "60+"])
    acc_by_age = compute_accuracy_by_group(df, "AgeGroup")
    acc_by_age.to_csv(os.path.join(output_dir, "accuracy_by_age.csv"))

    plt.figure(figsize=(6,5))
    sns.barplot(x=acc_by_age.index, y=acc_by_age.values)
    plt.title("Accuracy by Age Group")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_age.png"))
    plt.close()

    print("\n=== Bias Visualization Complete ===")
    print(f"Overall accuracy: {overall_acc:.3f}")
    print(f"Plots and CSV files saved to: {output_dir}")

if __name__ == "__main__":
    csv_path = "/Users/pc/Desktop/CODING/Others/fairvoice/evaluation_results/bias_report.csv"
    output_dir = "/Users/pc/Desktop/CODING/Others/fairvoice/evaluation_results/bias_charts"

    visualize_bias(csv_path, output_dir)
