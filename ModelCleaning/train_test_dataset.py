import os
from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Get the GitHub workspace path
workspace = os.getenv('GITHUB_WORKSPACE')

# Define the path to the ModelCleaning directory and cleaned data CSV file
model_cleaning_dir = os.path.join(workspace, 'ModelCleaning')
csv_file_path = os.path.join(model_cleaning_dir, 'Cleaned_Data.csv')  # Make sure this matches output from clean_data.py

# Debug: check file existence
if os.path.exists(csv_file_path):
    print(f"File found: {csv_file_path}")
else:
    print(f"File not found at: {csv_file_path}")

# Read the cleaned data
df = read_csv(csv_file_path)
print(df.head()) 

# Prepare features and labels
X = df["Age"].values.reshape(-1, 1)
y = df["Salary"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
mind = LinearRegression()
mind.fit(X_train, y_train)

# Save model for later use (optional)
model_path = os.path.join(model_cleaning_dir, 'linear_model.joblib')
dump(mind, model_path)

print(f"Model saved to: {model_path}")

