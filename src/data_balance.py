import pandas as pd

INPUT_CSV = '../labels/annotations.csv'
OUTPUT_CSV = '../labels/final_annotations.csv'
MAX_SAMPLES = 1500

def balance_dataset():
    annotations = pd.read_csv(INPUT_CSV)
    print("BEFORE BALANCE")
    print(annotations['command'].value_counts().sort_index())

    balanced_dfs = []

    for command_id in annotations['command'].unique():
        subset = annotations[annotations['command'] == command_id]
        count = len(subset)
        if count > MAX_SAMPLES:
            subset = subset.sample(n=MAX_SAMPLES, random_state=42)
        balanced_dfs.append(subset)

    final_df = pd.concat(balanced_dfs)

    final_df = final_df.sample(frac=1, random_state=42)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("AFTER BALANCE")
    print(final_df['command'].value_counts().sort_index())

if __name__ == "__main__":
    balance_dataset()