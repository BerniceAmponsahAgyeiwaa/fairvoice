# src/data/splitter.py
from pathlib import Path
import pandas as pd
import numpy as np
import json
import random
from collections import defaultdict, Counter

ROOT = Path("/Users/pc/Desktop/CODING/Others/fairvoice")
META_IN = ROOT / "data" / "processed" / "metadata.csv"
FEATURE_DIR = ROOT / "data" / "features"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

# target fractions
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

def extract_emotion(file_str: str):
    """
    Extract emotion code from filename pattern like '1001_IEO_HAP_LO'
    This function returns the 3rd token (index 2) if present; otherwise 'UNK'.
    """
    parts = str(file_str).split("_")
    if len(parts) >= 3:
        return parts[2]
    return "UNK"

def make_composite_demographic(row, fields=("Sex", "Race")):
    """
    Create a composite demographic label used for approximate stratification.
    Example: Sex='Male', Race='Caucasian' -> 'Male__Caucasian'
    """
    vals = []
    for f in fields:
        v = row.get(f, "")
        if pd.isna(v):
            v = "NA"
        vals.append(str(v))
    return "__".join(vals)

def speaker_actor_stats(df, demo_col, label_col):
    """
    Build a mapping actor -> counts per demographic label & emotion label.
    Returns:
      actor_list: list of actor IDs
      actor_info: dict actor -> {"n": int, "demo_counts": Counter, "label_counts": Counter}
    """
    actor_info = {}
    for actor, g in df.groupby("ActorID"):
        demo_counts = Counter(g[demo_col].values)
        label_counts = Counter(g[label_col].values)
        actor_info[actor] = {
            "n_files": len(g),
            "demo_counts": demo_counts,
            "label_counts": label_counts
        }
    return actor_info

def greedy_assign_actors(actor_info, demo_categories, target_fracs, rng):
    """
    Greedy assignment of actors to splits to approximately match target fractions
    across demographic categories.
    - actor_info: dict of actor->info
    - demo_categories: list of possible composite demographic values
    - target_fracs: dict {"train":0.7, "val":0.15, "test":0.15}
    Returns: dict actor -> split
    """
    # initialize counts per split per demographic
    splits = ["train", "val", "test"]
    split_demo_counts = {s: Counter() for s in splits}
    split_total = {s: 0 for s in splits}
    actor_list = list(actor_info.keys())
    rng.shuffle(actor_list)

    assignment = {}

    # helper to compute current distribution distance
    def distribution_distance_if_assigned(actor, split):
        # compute what demo counts would be if actor is assigned to split
        temp_counts = {s: split_demo_counts[s].copy() for s in splits}
        temp_totals = split_total.copy()
        for demo, c in actor_info[actor]["demo_counts"].items():
            temp_counts[split][demo] += c
            temp_totals[split] += c
        # compute L1 distance between each split's demographic fraction and target
        total_counts = sum(temp_totals.values())
        if total_counts == 0:
            return 0.0
        dist = 0.0
        # for each demographic category, compute desired proportion across splits
        # desired per split for a demo is target_fracs[split]
        for demo in demo_categories:
            # total occurrences of this demo in dataset if we use temp_counts
            tot_demo = sum(temp_counts[s][demo] for s in splits)
            if tot_demo == 0:
                continue
            for s in splits:
                current_frac = temp_counts[s][demo] / tot_demo
                dist += abs(current_frac - target_fracs[s])
        return dist

    # assign actors greedily
    for actor in actor_list:
        best_split = None
        best_score = None
        for s in splits:
            score = distribution_distance_if_assigned(actor, s)
            if best_score is None or score < best_score:
                best_score = score
                best_split = s
        # assign
        assignment[actor] = best_split
        # update counts
        for demo, c in actor_info[actor]["demo_counts"].items():
            split_demo_counts[best_split][demo] += c
            split_total[best_split] += c

    return assignment, split_demo_counts, split_total

def main():
    random.seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)

    if not META_IN.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_IN}")

    df = pd.read_csv(META_IN, sep=None, engine="python")
    # normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # ensure required columns exist
    required = {"file", "ActorID"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"metadata.csv must contain columns: {required}. Found: {df.columns.tolist()}")

    # 1) extract emotion label if not present in csv
    if "emotion" not in df.columns:
        df["emotion"] = df["file"].apply(extract_emotion)

    # 2) create composite demographic column for stratification
    # Using Sex and Race as required in your metadata; modify fields tuple if needed
    demo_fields = ("Sex", "Race")
    df["demo"] = df.apply(lambda r: make_composite_demographic(r, demo_fields), axis=1)

    # 3) check feature files exist (warn if not)
    def feature_exists(file_stem):
        feat_path = FEATURE_DIR / f"{file_stem}.pt"
        return feat_path.exists()

    df["feature_exists"] = df["file"].apply(lambda x: feature_exists(x))

    missing_feats = df[~df["feature_exists"]]
    if len(missing_feats) > 0:
        print(f"⚠️ WARNING: {len(missing_feats)} metadata rows have no matching feature .pt file in {FEATURE_DIR}.")
        print(missing_feats[["file", "clean_path"]].head(10).to_string(index=False))

    # 4) Build actor info and demographic categories
    actor_info = speaker_actor_stats(df, demo_col="demo", label_col="emotion")
    demo_categories = sorted({d for a in actor_info for d in actor_info[a]["demo_counts"].keys()})

    # 5) Greedy assign actors to splits
    target_fracs = {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC}
    assignment, split_demo_counts, split_total = greedy_assign_actors(actor_info, demo_categories, target_fracs, rng)

    # 6) annotate dataframe with split by ActorID
    df["split"] = df["ActorID"].map(assignment)

    # 7) Save split CSVs
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_fp = OUT_DIR / "metadata_train.csv"
    val_fp = OUT_DIR / "metadata_val.csv"
    test_fp = OUT_DIR / "metadata_test.csv"
    combined_fp = OUT_DIR / "metadata_splits.csv"

    train_df.to_csv(train_fp, index=False)
    val_df.to_csv(val_fp, index=False)
    test_df.to_csv(test_fp, index=False)
    df.to_csv(combined_fp, index=False)

    # 8) Print summary statistics
    def split_stats(dframe, name):
        n = len(dframe)
        by_demo = dframe["demo"].value_counts(normalize=True).head(10)
        by_emotion = dframe["emotion"].value_counts(normalize=True).head(10)
        return f"{name}: n={n}\n top demos:\n{by_demo.to_string()}\n top emotions:\n{by_emotion.to_string()}\n"

    print("\n=== Split summary ===\n")
    print(split_stats(train_df, "TRAIN"))
    print(split_stats(val_df, "VAL"))
    print(split_stats(test_df, "TEST"))

    # 9) Save assignment mapping for reproducibility
    assign_fp = OUT_DIR / "actor_split_assignment.json"
    with open(assign_fp, "w") as fh:
        json.dump(assignment, fh, indent=2)
    print(f"\n✅ Saved splits to:\n - {train_fp}\n - {val_fp}\n - {test_fp}\n - {combined_fp}")
    print(f"✅ Saved actor assignment mapping to: {assign_fp}")

if __name__ == "__main__":
    main()
