from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification
import torch

model_path = './model96.bin'

# Load the model with the specified configuration
model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base')
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

# Load the state dict from the saved model
model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Update the model with the loaded state dict
model.load_state_dict(model_state_dict, strict=False)

# Read Solidity smart contract file
contract_file_path = './malicious.sol'
with open(contract_file_path, 'r') as file:
    contract_code = file.read()

# Tokenize the contract code
tokens = tokenizer(contract_code, return_tensors='pt', max_length=2048, truncation=True)

# Make predictions using the RoBERTa model
with torch.no_grad():
    outputs = model(**tokens)

# Assuming you have a binary classification task (e.g., detect malicious/non-malicious contracts)
predictions = torch.argmax(outputs.logits, dim=1).item()

# Print the result
if predictions == 1:
    print("benign")
else:
    print("malicious")
