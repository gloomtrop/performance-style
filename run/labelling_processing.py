import pandas as pd

from utils.paths import path_from_root

# Values for scaling
max_value = 7
min_value = 0

# Save
SAVE = True

# Names and paths
file_name_input = 'data.csv'
file_name_output = 'data.json'
file_path_input = path_from_root('data', 'raw', 'labelling', file_name_input)
file_path_output = path_from_root('data', 'processed', 'labelling', file_name_output)
raw_df = pd.read_csv(file_path_input)

# Making index based on player, segment and user
player = raw_df['filename'].apply(lambda x: str(x.split('.')[0].split('_')[-2]))
segment = raw_df['filename'].apply(lambda x: str(x.split('.')[0].split('_')[-1]))
raw_df['id'] = player + '-' + segment + '-' + raw_df['user'].apply(lambda x: str(x))
raw_df = raw_df.set_index('id')

# Filtering question columns
data_columns = ['Question_1_1_1', 'Question_2_1_1',
                'Question_2_2_1', 'Question_3_1_1', 'Question_3_2_1', 'Question_3_3_1',
                'Question_4_1_1', 'Question_4_2_1', 'Question_4_3_1', 'Question_5_1_1',
                'Question_5_2_1', 'Question_5_3_1', 'Question_5_4_1', 'Question_5_5_1', 'Question_6_1_1',
                'Question_6_2_1', 'Question_6_3_1', 'Question_6_4_1',
                'Question_6_5_1', 'Question_7_1_1', 'Question_7_2_1', 'Question_7_3_1',
                'Question_8_1_1', 'Question_8_2_1', 'Question_9_1_1']
data = pd.DataFrame(raw_df[data_columns])

# Scaling
scaled_data = (data - min_value) / (max_value - min_value)

if SAVE:
    scaled_data.to_json(file_path_output)
