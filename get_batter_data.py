from pybaseball import playerid_lookup, statcast_batter
import pandas as pd

# Prompt user for batter name
first_name = input("Enter batter's first name: ").strip()
last_name = input("Enter batter's last name: ").strip()

# Step 1: Look up batter ID
batter_info = playerid_lookup(last_name, first_name)

if batter_info.empty:
    print(f"No player found for {first_name} {last_name}. Check spelling and try again.")
    exit()

batter_id = batter_info['key_mlbam'].values[0]
print(f"Using batter ID: {batter_id} for {first_name} {last_name}")

# Step 2: Pull Statcast data for 2023 season
print("Fetching pitch-by-pitch data for 2023...")
df = statcast_batter('2023-04-01', '2023-10-01', batter_id)

# Step 3: Save to CSV
filename = f"{last_name.lower()}_{first_name.lower()}_2023.csv"
df.to_csv(filename, index=False)
print(f"Saved data to {filename}")

# Step 4: Show preview
print(df[['pitch_type', 'description', 'events', 'release_speed', 'plate_x', 'plate_z']].head())
