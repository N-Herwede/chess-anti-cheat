import pandas as pd

df = pd.read_csv('/mnt/data/games_dataset_player_level_ultra_enriched.csv')

# On filtre seulement les lignes où la colonne "Moves" contient des évaluations [%eval ...]
filtered_df = df[df['Moves'].astype(str).str.contains(r'\[%eval', regex=True, na=False)]

filtered_df.to_csv('/mnt/data/with_eval_in_moves.csv', index=False)
print(filtered_df)
