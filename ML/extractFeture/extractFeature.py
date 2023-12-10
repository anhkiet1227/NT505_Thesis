from transformers import RobertaTokenizer
import os

def extract_token_ids_from_sol_file(file_path, model_name='roberta-base'):
    """
    Extracts token IDs from a .sol file using a specified RoBERTa model.

    Args:
    file_path (str): The path to the .sol file.
    model_name (str): The name of the RoBERTa model to be used for tokenization.

    Returns:
    List[int]: A list of token IDs.
    """
    # Ensure the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Initialize the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Read the .sol file
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()

    # Tokenize the file contents and convert to token IDs
    tokens = tokenizer.tokenize(file_contents)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return token_ids

# Example usage
file_path = 'Contract1.sol'
token_ids = extract_token_ids_from_sol_file(file_path)
print(token_ids)

# save the feature to csv file
#import pandas as pd
#df = pd.DataFrame(token_ids)
#df.to_csv('extractFeture.csv', index=False, header=False)

