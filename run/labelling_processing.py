import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from utils.paths import path_from_root

# Save
SAVE = True

# Names and paths
file_name_input = 'data.csv'
file_name_output = 'data.json'
file_path_input = path_from_root('data', 'raw', 'labelling', file_name_input)
file_path_output = path_from_root('data', 'processed', 'labelling', file_name_output)
raw_df = pd.read_csv(file_path_input)

# Filtering question columns
data_columns = ['Question_1_1_1', 'Question_2_1_1',
                'Question_2_2_1', 'Question_3_1_1', 'Question_3_2_1', 'Question_3_3_1',
                'Question_4_1_1', 'Question_4_2_1', 'Question_4_3_1', 'Question_5_1_1',
                'Question_5_2_1', 'Question_5_3_1', 'Question_5_4_1', 'Question_5_5_1', 'Question_6_1_1',
                'Question_6_2_1', 'Question_6_3_1', 'Question_6_4_1',
                'Question_6_5_1', 'Question_7_1_1', 'Question_7_2_1', 'Question_7_3_1',
                'Question_8_1_1', 'Question_8_2_1', 'Question_9_1_1']
data = pd.DataFrame(raw_df[data_columns])

data_np = data.to_numpy(dtype=float)
data_np[data_np == 0] = np.NAN

imp_mean = IterativeImputer(random_state=0)
imputed = imp_mean.fit_transform(data_np)
imputed_df = pd.DataFrame(imputed, columns=data.columns)

composer = raw_df['filename'].apply(lambda x: str(x.split('_')[0]))
piece = raw_df['filename'].apply(lambda x: str(x.split('_')[1]))
performer = raw_df['filename'].apply(lambda x: str(x.split('_')[3]))
segment = raw_df['filename'].apply(lambda x: str(x.split('.')[0].split('_')[-1]))

imputed_df['user'] = raw_df['user']
imputed_df['composer'] = composer
imputed_df['piece'] = piece
imputed_df['performer'] = performer
imputed_df['segment'] = segment

if SAVE:
    imputed_df.to_json(file_path_output)
