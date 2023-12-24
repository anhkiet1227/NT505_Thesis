import time
startTime = time.time()
import joblib
from transformers import RobertaTokenizer
import numpy as np

def detect_contract_type(token_ids, model_path, max_sequence_length=601):
    """
    Detects whether a contract is benign or malicious based on its token IDs.

    Args:
    token_ids (list): The token IDs of the contract.
    model_path (str): The path to the trained random forest model.
    max_sequence_length (int): The maximum sequence length expected by the model.

    Returns:
    str: The predicted contract type ('benign' or 'malicious').
    """
    # Ensure the input has the correct shape
    if len(token_ids) < max_sequence_length:
        # Pad token_ids if it is shorter than max_sequence_length
        token_ids = np.pad(token_ids, (0, max_sequence_length - len(token_ids)))
    elif len(token_ids) > max_sequence_length:
        # Truncate token_ids if it is longer than max_sequence_length
        token_ids = token_ids[:max_sequence_length]

    # Load the trained random forest model
    model = joblib.load(model_path)
    
    # Prepare the token IDs for the model
    inputs = np.array([token_ids])

    # Make a prediction
    predictions = model.predict(inputs)
    
    # Convert the prediction to a contract type
    contract_type = 'malicious' if predictions.item() == 0 else 'benign'

    return contract_type

# Read Solidity smart contract file
contract_file_path = './Contract2.sol'
with open(contract_file_path, 'r') as file:
    contract_code = file.read()

# Tokenize the contract code
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
tokens = tokenizer(contract_code, return_tensors="pt", truncation=True)
token_ids = tokens['input_ids'].squeeze().numpy()

# Detect the contract type using Random Forest
model_path = '../bin/random_forest.joblib'  # Replace with the path to your trained random forest model


contract_type = detect_contract_type(token_ids, model_path)


# Print the result and time taken
print(contract_type)

endTime = time.time()
print(f"Time taken: {endTime - startTime} seconds")
