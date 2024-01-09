import pandas as pd

# Read the .txt file with .csv version
input_file_path = 'dataset/SICK/SICK.txt'
df = pd.read_csv(input_file_path, sep='\t')  # Assuming tab-separated, change the separator if needed

# Split the data into three dataframes based on the "set" column
set1_df = df[df['SemEval_set'] == 'TRAIN']
set2_df = df[df['SemEval_set'] == 'TRIAL']
set3_df = df[df['SemEval_set'] == 'TEST']

# Specify the output file paths for the three .csv files
output_set1_path = 'dataset/SICK/train.csv'
output_set2_path = 'dataset/SICK/valid.csv'
output_set3_path = 'dataset/SICK/test.csv'

# Save the dataframes to the respective .csv files
set1_df.to_csv(output_set1_path, index=False)
set2_df.to_csv(output_set2_path, index=False)
set3_df.to_csv(output_set3_path, index=False)

print("Splitting and saving complete.")
