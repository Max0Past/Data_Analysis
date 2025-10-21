# Import the necessary libraries
import pandas as pd
from pathlib import Path

# --- Constructing the Path ---

# 1. Get the absolute path to the current script (test.py)
# Path(__file__) creates a Path object for the current file.
# .resolve() makes the path absolute
script_path = Path(__file__).resolve()

# 2. Get the directory containing the script (IAD/lab_3)
# .parent attribute gets the parent directory
script_dir = script_path.parent

# 3. Get the root directory (IAD)
# We go up one more level from script_dir (IAD/lab_3) to get IAD
iad_dir = script_dir.parent

# 4. Construct the path to the data directory (IAD/lab_2/data)
# The '/' operator is overloaded by pathlib to join paths correctly
data_dir = iad_dir / 'lab_2' / 'data'

# --- Loading the Dataset ---

# 5. Specify your dataset name
dataset_name = 'avocado.csv'

# 6. Create the full file path
file_path = data_dir / dataset_name

# 7. Load the data (example using pandas)
try:
    print(f"Attempting to load data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Successfully loaded
    print("Data loaded successfully.")
    print(df.head())

except FileNotFoundError:
    print(f"ERROR: File not found at the specified path.")
    print(f"Please check if '{dataset_name}' exists in '{data_dir}'")
except Exception as e:
    print(f"An error occurred: {e}")