import csv

# Define the ranges for benign and malicious contracts
benign_range = (1, 158)
reentrancy_range = (159, 200)
integer_overflow_range = (201, 248)
unprotected_withdrawal_range = (249, 300)

# Create a list to store the contract labels with details
contract_labels = []

# Label the contracts
for contract_num in range(1, 301):
    if benign_range[0] <= contract_num <= benign_range[1]:
        label = "benign"
        detail = "N/A"
    elif reentrancy_range[0] <= contract_num <= reentrancy_range[1]:
        label = "malicious"
        detail = "Reentrancy"
    elif integer_overflow_range[0] <= contract_num <= integer_overflow_range[1]:
        label = "malicious"
        detail = "Integer Overflow/Underflow"
    elif unprotected_withdrawal_range[0] <= contract_num <= unprotected_withdrawal_range[1]:
        label = "malicious"
        detail = "Unprotected Ether Withdrawal"
    else:
        label = "unknown"
        detail = "N/A"
    
    # Append the contract number, label, and detail to the list
    contract_labels.append((f"Contract{contract_num}", label, detail))

# Define the CSV file path
csv_file_path = "detail_label.csv"

# Write the contract labels to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Contract', 'Label', 'Detail'])  # Write header
    writer.writerows(contract_labels)  # Write data

print(f"Labels with details written to {csv_file_path}.")
