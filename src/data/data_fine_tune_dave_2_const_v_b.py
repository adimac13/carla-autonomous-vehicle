#Model based on NVIDIA's DAVE-2 with constant throttle=0.18 and brake=0.0
import pandas as pd
import numpy as np

INPUT_CSV = '../../labels/dave2_const_v_s/annotations.csv'
OUTPUT_CSV = '../../labels/dave2_const_v_s/fine_tuning_annotations.csv'


NEAR_ZERO_RANGE = 0.15
MAX_SAMPLES_STEER_NEAR_0 = 2300
MAX_SAMPLES_STEER_OTHER = 2000

MAX_SAMPLES_STRAIGHT_STEER_TURN = 500

MAX_SAMPLES_COMMAND12 = 200
MAX_SAMPLES_COMMAND3 = 8000
MAX_SAMPLES_COMMAND4 = 1500

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

    # bins = np.arange(-1.0, 1.01, 0.05)
    # subset_command_3['range'] = pd.cut(subset_command_3['steer'], bins=bins)
    # print(subset_command_3['range'].value_counts().sort_index())

    subset_command_3_straight = subset_command_3[abs(subset_command_3['steer']) <= 0.05]
    subset_command_3_slight_turn = subset_command_3[abs(subset_command_3['steer']) > 0.05].sample(n = MAX_SAMPLES_STRAIGHT_STEER_TURN, random_state = 42)
    subset_command_3 = pd.concat([subset_command_3_straight, subset_command_3_slight_turn])

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


if __name__ == "__main__":
    balance_dataset_method_2()