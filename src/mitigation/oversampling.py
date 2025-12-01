import os
import pandas as pd
from sklearn.utils import resample

def oversample_by_race(metadata_csv, output_csv):
    df = pd.read_csv(metadata_csv)

    groups = df.groupby("Race")
    max_size = groups.size().max()

    oversampled = []

    for race, group in groups:
        if len(group) < max_size:
            print(f"Oversampling {race}: {len(group)} → {max_size}")
            group_over = resample(
                group,
                replace=True,
                n_samples=max_size,
                random_state=42
            )
            oversampled.append(group_over)
        else:
            oversampled.append(group)

    final_df = pd.concat(oversampled)
    final_df.to_csv(output_csv, index=False)

    print(f"Oversampled metadata saved to {output_csv}")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    meta_path = os.path.join(project_root, "data/processed/metadata.csv")
    out_path = os.path.join(project_root, "data/processed/metadata_oversampled.csv")

    oversample_by_race(meta_path, out_path)
