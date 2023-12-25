from transformers import RobertaTokenizer

def tokenize_contract(file_path, model_name='roberta-base'):
    """
    Tokenizes a smart contract and prints the tokens and token IDs.

    Args:
    file_path (str): The path to the smart contract file.
    model_name (str): The name of the RoBERTa model to be used for tokenization.
    """
    # Initialize the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Tokenize and convert to token IDs
    tokens = tokenizer.tokenize(content)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Print the tokens and token IDs
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")

# Example usage
file_path = './Contract1.sol'
tokenize_contract(file_path)