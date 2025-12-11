import os
import pandas as pd

# Set the folder path
folder_path = './reports'

# List to hold each DataFrame
df_list = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, sep=';')
        df_list.append(df)

# Combine all DataFrames
combined_df = pd.concat(df_list, ignore_index=True)
#combined_df['iteration'] = combined_df['Total_Steps'] / 4096
#combined_df['iteration'] = combined_df['iteration'].astype(int)
combined_df['total_damage_dealt'] = combined_df['damage_0'] + combined_df['damage_0'] + combined_df['damage_1'] + combined_df['damage_2'] + combined_df['damage_3'] + combined_df['damage_4'] + combined_df['damage_5'] + combined_df['damage_6'] + combined_df['damage_7']
# Optional: save to a new CSV

action_names = {
    'action_0': 'Up',
    'action_1': 'Down',
    'action_2': 'Left',
    'action_3': 'Right',
    'action_4': 'Punch (A)',
    'action_5': 'Kick (B)',
    'action_6': 'Hurricane Kick',
    'action_7': 'Hadouken',
    'damage_0': 'Damage Up',
    'damage_1': 'Damage Down',
    'damage_2': 'Damage Left',
    'damage_3': 'Damage Right',
    'damage_4': 'Damage Punch',
    'damage_5': 'Damage Kick',
    'damage_6': 'Damage Hurricane',
    'damage_7': 'Damage Hadouken'
}


combined_df = combined_df.rename(columns=action_names)


combined_df.to_csv('combined_output.csv', index=False, sep=';')

# If you want to just return the DataFrame
print(combined_df)