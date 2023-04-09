import os
import pickle

# Navigate to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Define the path to the pickle file in the parent directory
pickle_path = os.path.join(parent_dir, "example.pickle")

# Load the data from the pickle file
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# Print the loaded data
print(data)
