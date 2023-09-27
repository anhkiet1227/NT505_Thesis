import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load the dataset
dataset_file = 'synthetic_smart_contract_dataset.csv'
data = pd.read_csv(dataset_file)

# Features (X) and labels (y)
X = data['Smart Contract Code']
y = data['Reentrancy Vulnerability']  # Change this to the vulnerability you want to detect

# Binarize the labels
y = y.apply(lambda x: 1 if x == 'Yes' else 0)  # Assuming 'Yes' indicates vulnerability, adjust if needed

# Tokenize the text using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_inputs = tokenizer(X.tolist(), padding='max_length', truncation=True, return_tensors='pt')

# Split the data into training and testing sets
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']
X_train, X_test, y_train, y_test = train_test_split(input_ids, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
y_train = torch.tensor(y_train.values)
y_test = torch.tensor(y_test.values)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train, attention_mask[:len(X_train)], y_train)
test_dataset = TensorDataset(X_test, attention_mask[len(X_train):], y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(5):  # Adjust the number of epochs as needed
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, masks, labels = batch
        outputs = model(inputs, attention_mask=masks)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, masks, labels = batch
        outputs = model(inputs, attention_mask=masks)[0]
        _, predicted_labels = torch.max(outputs, 1)
        predictions.extend(predicted_labels.tolist())
        true_labels.extend(labels.tolist())

# Evaluate the model
print('Accuracy:', accuracy_score(true_labels, predictions))
print('Classification Report:\n', classification_report(true_labels, predictions))
print('Confusion Matrix:\n', confusion_matrix(true_labels, predictions))
