from transformers import RobertaTokenizer
import csv
import os
import re

def tokenize_contracts_in_folder_and_write_to_csv_with_labels(folder_path, csv_file_path, model_name='roberta-base'):
    """
    Tokenizes smart contracts in a folder and writes the token IDs and labels to a CSV file based on file name.

    Args:
    folder_path (str): The path to the folder containing the smart contracts.
    csv_file_path (str): The path to the CSV file where results will be written.
    model_name (str): The name of the RoBERTa model to be used for tokenization.
    """
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    # Initialize the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Prepare to write to the CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File Name', 'Token IDs', 'Label'])  # Writing header

        # Process each file in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Tokenize and convert to token IDs
            tokens = tokenizer.tokenize(content)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Extract number from filename and determine the label
            match = re.search(r'\d+', filename)
            if match:
                file_number = int(match.group())
                label = "benign" if 1 <= file_number <= 158 else "malicious"
            else:
                label = "unknown"

            # Write to CSV
            writer.writerow([filename, token_ids, label])

# Example usage
folder_path = './Total'
output_csv_file_path = 'output_with_labels.csv'
tokenize_contracts_in_folder_and_write_to_csv_with_labels(folder_path, output_csv_file_path)
