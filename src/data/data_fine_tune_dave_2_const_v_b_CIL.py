#Model based on NVIDIA's DAVE-2 and Conditional Imitation Learning with constant throttle=0.18 and brake=0.0
import pandas as pd
import numpy as np

INPUT_CSV = '../../labels/dave2_const_v_s/annotations.csv'
OUTPUT_CSV = '../../labels/dave2_const_v_s/fine_tuning_annotations.csv'


NEAR_ZERO_RANGE = 0.15
MAX_SAMPLES_STEER_NEAR_0 = 2300
MAX_SAMPLES_STEER_OTHER = 2000

MAX_SAMPLES_STRAIGHT_STEER_TURN = 1200

MAX_SAMPLES_COMMAND12 = 0
MAX_SAMPLES_COMMAND3 = 0
MAX_SAMPLES_COMMAND4 = 4000

#New method concentrates on reducing data where steer in range [0,0.1] while following a lane
#The rest of the data is only limited by the value of MAX_SAMPLES_COMMANDx
def balance_dataset_method():
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

    if len(subset_command_3) > MAX_SAMPLES_COMMAND3:
        subset_command_3 = subset_command_3.sample(n = MAX_SAMPLES_COMMAND3, random_state = 42)

    subset_command_4_straight = subset_command_4[abs(subset_command_4['steer']) <= 0.11].sample(
        n=MAX_SAMPLES_STRAIGHT_STEER_TURN, random_state=42)
    subset_command_4_turn = subset_command_4[abs(subset_command_4['steer']) > 0.11]
    subset_command_4 = pd.concat([subset_command_4_straight, subset_command_4_turn])

    if len(subset_command_4) > MAX_SAMPLES_COMMAND4:
        subset_command_4 = subset_command_4.sample(n=MAX_SAMPLES_COMMAND4, random_state=42)

    final_df = pd.concat([subset_command_1, subset_command_2, subset_command_3,subset_command_4])
    final_df = final_df.sample(frac = 1, random_state = 42)

    final_df.to_csv(OUTPUT_CSV, index=False)

    print("AFTER BALANCE")
    print(final_df['command'].value_counts().sort_index())


if __name__ == "__main__":
    balance_dataset_method()