import csv
import random
import string

# Function to generate a random smart contract code (placeholder in this example)
def generate_contract_code():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

# Function to generate a balanced mix of vulnerability labels (70% with vulnerabilities, 30% without)
def generate_vulnerability_label():
    return random.choices([0, 1], weights=[0.2756, 0.7244])[0]

# Number of synthetic contract entries in the dataset
num_entries = 101375

# Header for the CSV file
header = ['Contract Name', 'Smart Contract Code', 'Gas Usage', 'Function Calls', 'External Calls',
          'Reentrancy Vulnerability', 'Overflow/Underflow Vulnerability', 'Unprotected Ether Withdrawal']

# Generate synthetic data and write to CSV file
with open('synthetic_smart_contract_dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for i in range(num_entries):
        contract_name = f'Contract{i+1}'
        contract_code = generate_contract_code()
        gas_usage = random.randint(10000, 50000)
        function_calls = random.randint(5, 20)
        external_calls = random.randint(2, 10)
        reentrancy_vulnerability = generate_vulnerability_label()
        overflow_underflow_vulnerability = generate_vulnerability_label()
        unprotected_withdrawal_vulnerability = generate_vulnerability_label()

        writer.writerow([contract_name, contract_code, gas_usage, function_calls, external_calls,
                         reentrancy_vulnerability, overflow_underflow_vulnerability, unprotected_withdrawal_vulnerability])

print('Synthetic dataset created and saved to synthetic_smart_contract_dataset.csv.')
