#Model based on NVIDIA's DAVE-2 with constant throttle=0.18 and brake=0.0
#TODO Change data - harsher steer on imageX_right
import pandas as pd
import numpy as np

INPUT_CSV = '../../labels/dave2_const_v_s/annotations.csv'
OUTPUT_CSV = '../../labels/dave2_const_v_s/final_annotations.csv'

#By far the biggest improvement when
# NEAR_ZERO_RANGE = 0.15
# MAX_SAMPLES_STEER_NEAR_0 = 2500
# MAX_SAMPLES_STEER_OTHER = 3000
#
# MAX_SAMPLES_COMMAND12 = 2800
# MAX_SAMPLES_COMMAND3 = 3500
# MAX_SAMPLES_COMMAND4 = 3500
#
#Data distribution
# 1    2800
# 2    2800
# 3    2380
# 4    3500

NEAR_ZERO_RANGE = 0.15
MAX_SAMPLES_STEER_NEAR_0 = 3000
MAX_SAMPLES_STEER_OTHER = 2000

MAX_SAMPLES_COMMAND12 = 8000
MAX_SAMPLES_COMMAND3 = 7000
MAX_SAMPLES_COMMAND4 = 11000

#New method concentrates on reducing data where steer in range [0,0.1] while following a lane
#The rest of the data is only limited by the value of MAX_SAMPLES_COMMANDx
def balance_dataset_method_2():
    annotations = pd.read_csv(INPUT_CSV)
    subset_command_1 = annotations[annotations['command'] == 1].copy()
    subset_command_2 = annotations[annotations['command'] == 2].copy()
    subset_command_3 = annotations[annotations['command'] == 3].copy()
    subset_command_4 = annotations[annotations['command'] == 4].copy()

    if len(subset_command_1) > MAX_SAMPLES_COMMAND12:
        subset_command_1 = subset_command_1.sample(n = MAX_SAMPLES_COMMAND12, random_state = 42)

    if len(subset_command_2) > MAX_SAMPLES_COMMAND12:
        subset_command_2 = subset_command_2.sample(n = MAX_SAMPLES_COMMAND12, random_state = 42)

    if len(subset_command_3) > MAX_SAMPLES_COMMAND3:
        subset_command_3 = subset_command_3.sample(n = MAX_SAMPLES_COMMAND3, random_state = 42)

    if len(subset_command_4) > MAX_SAMPLES_COMMAND4:
        bins = np.arange(-1.0, 1.01, 0.05)
        subset_command_4 ['range'] = pd.cut(subset_command_4['steer'], bins = bins)
        # print(subset_command_4['range'].value_counts().sort_index())
        balanced_4_stage_1 = []
        for range_id in subset_command_4['range'].unique():
            subset = subset_command_4[subset_command_4['range'] == range_id]

            count = len(subset)
            left = abs(float(range_id.left))
            right = abs(float(range_id.right))
            max_abs_range = np.maximum(left, right)

            if max_abs_range < NEAR_ZERO_RANGE:
                if count > MAX_SAMPLES_STEER_NEAR_0:
                    subset = subset.sample(n = MAX_SAMPLES_STEER_NEAR_0, random_state=42)
            else:
                if count > MAX_SAMPLES_STEER_OTHER:
                    subset = subset.sample(n = MAX_SAMPLES_STEER_OTHER, random_state=42)
            balanced_4_stage_1.append(subset)

        balanced_4_stage_2 = pd.concat(balanced_4_stage_1)
        balanced_4_stage_2 = balanced_4_stage_2.drop(columns='range')

        if len(balanced_4_stage_2) > MAX_SAMPLES_COMMAND4:
            balanced_4_stage_2 = balanced_4_stage_2.sample(n = MAX_SAMPLES_COMMAND4, random_state=42)

        subset_command_4 = balanced_4_stage_2

    final_df = pd.concat([subset_command_1, subset_command_2, subset_command_3,subset_command_4])
    final_df = final_df.sample(frac = 1, random_state = 42)

    final_df.to_csv(OUTPUT_CSV, index=False)

    print("AFTER BALANCE")
    print(final_df['command'].value_counts().sort_index())

#
def balance_dataset_method_1():
    annotations = pd.read_csv(INPUT_CSV)
    print("BEFORE BALANCE")
    print(annotations['command'].value_counts().sort_index())

    print("BEFORE BALANCE")
    bins = np.arange(-1.0, 1.01, 0.05)
    annotations ['range'] = pd.cut(annotations['steer'], bins = bins)
    print(annotations['range'].value_counts().sort_index())

    balanced_annotations = []

    #Balancing annotations by the value of steer
    for range_id in annotations['range'].unique():
        subset = annotations[annotations['range'] == range_id]

        count = len(subset)
        left = abs(float(range_id.left))
        right = abs(float(range_id.right))
        max_abs_range = np.maximum(left, right)

        if max_abs_range < NEAR_ZERO_RANGE:
            if count > MAX_SAMPLES_STEER_NEAR_0:
                subset = subset.sample(n=MAX_SAMPLES_STEER_NEAR_0, random_state=42)
        else:
            if count > MAX_SAMPLES_STEER_OTHER:
                subset = subset.sample(n=MAX_SAMPLES_STEER_OTHER, random_state=42)
        balanced_annotations.append(subset)


    mid_annotations = pd.concat(balanced_annotations)
    mid_annotations = mid_annotations.drop(columns=['range'])

    new_balanced_annotations = []

    #Balancing new annotations by the value of command
    for command_id in mid_annotations['command'].unique():
        subset = mid_annotations[mid_annotations['command'] == command_id]
        count = len(subset)
        if command_id == 4:
            if count > MAX_SAMPLES_COMMAND4:
                subset = subset.sample(n=MAX_SAMPLES_COMMAND4, random_state=42)
        elif command_id == 3:
            if count > MAX_SAMPLES_COMMAND3:
                subset = subset.sample(n=MAX_SAMPLES_COMMAND3, random_state=42)
        else:
            if count > MAX_SAMPLES_COMMAND12:
                subset = subset.sample(n=MAX_SAMPLES_COMMAND12, random_state=42)
        new_balanced_annotations.append(subset)

    final_df = pd.concat(new_balanced_annotations)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("AFTER BALANCE")
    print(final_df['command'].value_counts().sort_index())


if __name__ == "__main__":
    balance_dataset_method_2()