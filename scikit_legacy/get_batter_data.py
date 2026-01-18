from pybaseball import playerid_lookup, statcast_batter
import pandas as pd

# Prompt user for batter name
first_name = input("Enter batter's first name: ").strip()
last_name = input("Enter batter's last name: ").strip()

# Look up batter ID
batter_info = playerid_lookup(last_name, first_name)

if batter_info.empty:
    print(f"No player found for {first_name} {last_name}. Check spelling and try again.")
    exit()

batter_id = batter_info['key_mlbam'].values[0]
print(f"Using batter ID: {batter_id} for {first_name} {last_name}")

#Pull Statcast data for 2023 season
print("Fetching pitch-by-pitch data for 2023...")
df = statcast_batter('2023-04-01', '2023-10-01', batter_id)

# Keep only useful columns
keep_columns = [
    'game_date', 'pitch_type', 'pitch_name', 'release_speed',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'zone',
    'description', 'events', 'balls', 'strikes', 'outs_when_up',
    'inning', 'stand', 'p_throws', 'launch_speed', 'launch_angle',
    'hit_distance_sc', 'home_team', 'away_team'
]

df_cleaned = df[keep_columns]


# Save cleaned data
filename = f"{last_name.lower()}_{first_name.lower()}_2023_cleaned.csv"
df_cleaned.to_csv(filename, index=False)
print(f"Saved cleaned data to {filename}")


# Save to CSV
filename = f"{last_name.lower()}_{first_name.lower()}_2023.csv"
df.to_csv(filename, index=False)
print(f"Saved data to {filename}")


#save filename (Batter Name)
with open("last_batter_filename.txt", "w") as f:
    f.write(filename)

# Show preview
print(df_cleaned.head())

df = pd.read_csv(filename)

# label effective pitches
'''
effective_description = [
    'swinging_strike',
    'swingin_strike_blocked';
    'called_strike'
]
'''