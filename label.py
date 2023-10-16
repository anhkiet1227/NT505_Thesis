import csv

# Define the ranges for benign and malicious contracts
benign_range = (1, 158)
malicious_range = (159, 300)

# Create a list to store the contract labels
contract_labels = []

# Label the contracts
for contract_num in range(1, 301):
    if benign_range[0] <= contract_num <= benign_range[1]:
        label = "benign"
    elif malicious_range[0] <= contract_num <= malicious_range[1]:
        label = "malicious"
    else:
        label = "unknown"
    
    # Append the contract number and label to the list
    contract_labels.append((f"Contract{contract_num}", label))

# Define the CSV file path
csv_file_path = "contract_labels.csv"

# Write the contract labels to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Contract', 'Label'])  # Write header
    writer.writerows(contract_labels)  # Write data

print(f"Labels written to {csv_file_path}.")
