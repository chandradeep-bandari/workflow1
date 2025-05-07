import pandas as pd

df = pd.read_csv("ModelCleaning/Data.csv")

# Basic cleaning
df.columns = df.columns.str.strip()  # Trim column headers
df['Name'] = df['Name'].str.strip().str.title()
df['Email'] = df['Email'].str.strip().str.lower()

# Drop rows with missing names or ages
df = df.dropna(subset=["Name", "Age"])

# Save cleaned version
df.to_csv("ModelCleaning/Cleaned_Data.csv", index=False)

print("Data cleaned and saved to ModelCleaning/Cleaned_Data.csv")

