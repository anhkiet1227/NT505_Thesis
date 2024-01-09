from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the data, encode labels, pad sequences, and split the data
# ... (same as your previous code)
# Load the data
df = pd.read_csv('output_with_labels.csv')
labels = df['Label']
token_ids = df['Token IDs'].apply(eval)  # Assuming token IDs are stored as strings of lists

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Pad sequences for LSTM input
max_length = max(len(tokens) for tokens in token_ids)
padded_token_ids = pad_sequences(token_ids, maxlen=max_length, padding='post')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(padded_token_ids, encoded_labels, test_size=0.2, random_state=42)
# Load the data, encode labels, pad sequences, and split the data
# ... (same as your previous code)

# Create the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the model
model.fit(X_train, y_train)

# save the model
import joblib
joblib.dump(model, '../bin/xgboost.joblib')

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1: {f1_score(y_test, y_pred) * 100:.2f}%")

# Print classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))