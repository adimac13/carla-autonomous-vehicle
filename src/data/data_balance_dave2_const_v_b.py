import pandas as pd
import numpy as np

INPUT_CSV = '../../labels/dave2_const_v_s/annotations.csv'
OUTPUT_CSV = '../../labels/dave2_const_v_s/final_annotations.csv'

#By far the biggest improvement when
# MAX_SAMPLES_STEER_NEAR_0 = 400
# MAX_SAMPLES_STEER_OTHER = 800
#
# MAX_SAMPLES_COMMAND4 = 2000
# MAX_SAMPLES_COMMAND123 = 2100

MAX_SAMPLES_STEER_NEAR_0 = 700
MAX_SAMPLES_STEER_OTHER = 1000

MAX_SAMPLES_COMMAND4 = 3000
MAX_SAMPLES_COMMAND123 = 3000

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
        left = abs(float(range_id.left))
        right = abs(float(range_id.right))
        max_abs_range = np.maximum(left, right)

        if max_abs_range < 0.25:
            if count > MAX_SAMPLES_STEER_NEAR_0:
                subset = subset.sample(n=MAX_SAMPLES_STEER_NEAR_0, random_state=42)
        else:
            if count > MAX_SAMPLES_STEER_OTHER:
                subset = subset.sample(n=MAX_SAMPLES_STEER_OTHER, random_state=42)
        balanced_annotations.append(subset)


    mid_annotations = pd.concat(balanced_annotations)
    mid_annotations = mid_annotations.drop(columns=['range'])

    new_balanced_annotations = []

    for command_id in mid_annotations['command'].unique():
        subset = mid_annotations[mid_annotations['command'] == command_id]
        count = len(subset)
        if command_id == 4:
            if count > MAX_SAMPLES_COMMAND4:
                subset = subset.sample(n=MAX_SAMPLES_COMMAND4, random_state=42)
        else:
            if count > MAX_SAMPLES_COMMAND123:
                subset = subset.sample(n=MAX_SAMPLES_COMMAND123, random_state=42)
        new_balanced_annotations.append(subset)

    final_df = pd.concat(new_balanced_annotations)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("AFTER BALANCE")
    print(final_df['command'].value_counts().sort_index())


if __name__ == "__main__":
    balance_dataset()