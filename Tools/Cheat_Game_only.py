import pandas as pd

df = pd.read_csv('/mnt/data/small_games_dataset.csv')
filtered_df = df[df['TOS violation']
                 .astype(str)
                 .str.strip()
                 .str.lower()
                 .isin(['true', '1'])]
filtered_df.to_csv('/mnt/data/tos_violation_only.csv', index=False)
print(filtered_df)
