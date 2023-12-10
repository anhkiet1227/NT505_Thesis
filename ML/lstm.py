import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

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

# Create the LSTM model
model = Sequential()
model.add(Embedding(input_dim=50265, output_dim=64, input_length=max_length))  # 50265 is the size of RoBERTa-base vocab
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype('int32')  # Convert probabilities to binary predictions

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate and print classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)