import pandas as pd
print("Hello World")
# Step 1: Load the filename from the shared file
try:
    with open("last_batter_filename.txt", "r") as f:
        input_file = f.read().strip()
except FileNotFoundError:
    print("Shared filename file not found. Please run the data fetch script first.")
    exit()

# Step 2: Load the CSV
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"File '{input_file}' not found.")
    exit()

# Step 3: Label "effective" pitches
effective_descriptions = [
    'swinging_strike',
    'swinging_strike_blocked',
    'called_strike'
]

df['effective'] = df['description'].isin(effective_descriptions).astype(int)

# Also count foul balls with 2 strikes as effective
df.loc[(df['description'] == 'foul') & (df['strikes'] == 2), 'effective'] = 1

# Step 4: Save the labeled version
output_file = input_file.replace('_cleaned.csv', '_labeled.csv')
df.to_csv(output_file, index=False)

# Step 5: Show preview
print(f"Labeled data saved to '{output_file}'")
print(df[['pitch_type', 'description', 'strikes', 'effective']].head())
