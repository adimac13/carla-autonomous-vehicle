from data_balance_dave2 import balance_dataset

INPUT_CSV = '../../labels/dave2_const_v_s/annotations.csv'
OUTPUT_CSV = '../../labels/dave2_const_v_s/final_annotations.csv'
MAX_SAMPLES_STEER = 1000
MAX_SAMPLES_COMMAND4 = 700
MAX_SAMPLES_COMMAND123 = 2200

if __name__ == "__main__":
    balance_dataset(INPUT_CSV = INPUT_CSV, OUTPUT_CSV = OUTPUT_CSV,
                    MAX_SAMPLES_STEER = MAX_SAMPLES_STEER, MAX_SAMPLES_COMMAND4 = MAX_SAMPLES_COMMAND4, MAX_SAMPLES_COMMAND123 = MAX_SAMPLES_COMMAND123)