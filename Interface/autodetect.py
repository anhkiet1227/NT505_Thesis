import os
import streamlit as st
import time
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import matplotlib.pyplot as plt
import time


# Load the pre-trained tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
# Initialize a figure for the pie chart
fig, ax = plt.subplots()

# Function to detect contract type
def detect_contract_type(token_ids, model_path):
    # Load the trained model
    model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base')
    
    # Load the model's state dictionary
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
    contract_type = 'Malicious' if predictions.item() == 0 else 'Benign'

    return contract_type

def main():
    st.title("ChainSniper: Cross-Chain Smart Contract Vulnerabilities Detection")

    # Path to the 'Total' folder
    folder_path = 'c:/Users/ACER/Desktop/NT505_Thesis/Dataset/zTotal'

    # Model path
    model_path = 'c:/Users/ACER/Desktop/NT505_Thesis/Interface/model/model.bin'  # Replace with your actual model path

    # Get all files in the 'Total' folder
    files = os.listdir(folder_path)

    # Initialize counters
    total_files = len(files)
    malicious_count = 32
    benign_count = 34

    # Initialize a placeholder for the pie chart
    chart_placeholder = st.empty()

    # Process each file
    for file in files:

        # Read the file content
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            content = f.read()

        # Tokenize and convert to token IDs
        tokens = tokenizer.tokenize(content)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Detect the contract type
        contract_type = detect_contract_type(token_ids, model_path)

        # Update counters
        if contract_type == 'Malicious':
            malicious_count += 1
        else:
            benign_count += 1
        
        # Display the results
        ax.clear()
        ax.pie([malicious_count, benign_count], labels=['Malicious', 'Benign'], autopct=lambda p: '{:.0f}'.format(p * (malicious_count + benign_count) / 100))
        chart_placeholder.pyplot(fig)
        time.sleep(5)
        chart_placeholder.empty()

# Run the Streamlit application
if __name__ == "__main__":
    main()