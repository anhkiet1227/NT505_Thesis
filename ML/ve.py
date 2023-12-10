import time
startTime = time.time()

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch


def detect_contract_type(token_ids, model_path):
    """
    Detects whether a contract is benign or malicious based on its token IDs.

    Args:
    token_ids (list): The token IDs of the contract.
    model_path (str): The path to the trained model.

    Returns:
    str: The predicted contract type ('benign' or 'malicious').
    """
    # Load the trained model
    model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base')
    #tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # Update the model with the loaded state dict
    model.load_state_dict(model_state_dict, strict=False)
    # Prepare the token IDs for the model
    inputs = torch.tensor([token_ids])

    # Make a prediction
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Convert the prediction to a contract type
    contract_type = 'malicious' if predictions.item() == 0 else 'benign'

    return contract_type



# Read Solidity smart contract file
contract_file_path = './benign.sol'
with open(contract_file_path, 'r') as file:
    contract_code = file.read()

# Tokenize the contract code
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
tokens = tokenizer(contract_code)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Detect the contract type
contract_type = detect_contract_type(token_ids, model_path='./model/model.bin')

# Print the result
print(contract_type)
endTime = time.time()
print(endTime-startTime)