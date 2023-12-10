import json
import os

# Define the folder path
folder_path = './Total/'

# Initialize an empty list to store the results
results = []

# Iterate through the files in the folder
for idx in range(1, 301):
    file_name = f'Contract{idx}.sol'
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = file.read().rstrip()

        # Determine the 'target' based on the file number
        target = 0 if idx <= 158 else 1

        # Create a dictionary for the current contract
        contract_data = {
            'project': file_name,
            'commit_id': '0',
            'target': target,
            'func': data,
            'idx': str(idx)
        }

        results.append(contract_data)

# Write the results to a JSON file
with open('result.json', 'w') as fp:
    json.dump(results, fp)
