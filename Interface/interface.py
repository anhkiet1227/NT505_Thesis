import streamlit as st
import time
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load the pre-trained tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

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

# Streamlit UI
def main():
    st.title("ChainSniper: Cross-Chain Smart Contract Vulnerabilities Detection")

    # Input area for the smart contract content
    contract_content = st.text_area("Enter Smart Contract Code")

    if st.button("Detect Contract"):
        # Tokenize the contract code
        tokens = tokenizer(contract_content)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Detect the contract type
        contract_type = detect_contract_type(token_ids, model_path='c:/Users/ACER/Desktop/NT505_Thesis/Interface/model/model.bin')

        # Display the result
        st.success(f"Contract Status: {contract_type}")
    
    uploaded_file = st.file_uploader("Upload a .sol file", type=["sol"])

    if uploaded_file is not None:
        # Read the content of the uploaded file
        contract_content = uploaded_file.read().decode("utf-8")

        # Display the content of the uploaded file
        st.subheader("Uploaded Smart Contract Code:")
        st.code(contract_content, language="solidity")

        if st.button("Detect Contract Type"):
            # Tokenize the contract code
            tokens = tokenizer(contract_content)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Detect the contract type
            contract_type = detect_contract_type(token_ids, model_path='c:/Users/ACER/Desktop/NT505_Thesis/Interface/model/model.bin')

            # Display the result
            st.success(f"Contract Type: {contract_type}")


if __name__ == "__main__":
    main()
