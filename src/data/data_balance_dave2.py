import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = '../../labels/dave2/annotations.csv'
OUTPUT_CSV = '../../labels/dave2/final_annotations.csv'
MAX_SAMPLES = 400

def balance_dataset():
    annotations = pd.read_csv(INPUT_CSV)
    print("BEFORE BALANCE")
    bins = np.arange(-1.0, 1.01, 0.05)
    annotations ['range'] = pd.cut(annotations['steer'], bins = bins)
    print(annotations['range'].value_counts().sort_index())


    balanced_annotations = []

    for range_id in annotations['range'].unique():
        subset = annotations[annotations['range'] == range_id]
        count = len(subset)
        if count > MAX_SAMPLES:
            subset = subset.sample(n=MAX_SAMPLES, random_state=42)
        balanced_annotations.append(subset)



    final_df = pd.concat(balanced_annotations)

    final_df = final_df.drop(columns=['range'])
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("AFTER BALANCE")
    print(final_df['command'].value_counts().sort_index())

if __name__ == "__main__":
    balance_dataset()