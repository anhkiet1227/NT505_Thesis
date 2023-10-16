import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.utils import to_categorical

# Load the dataset
dataset_file = 'synthetic_smart_contract_dataset.csv'
data = pd.read_csv(dataset_file)

# Features (X) and labels (y)
X = data.drop(['Contract Name', 'Smart Contract Code', 'Reentrancy Vulnerability', 'Overflow/Underflow Vulnerability', 'Unprotected Ether Withdrawal'], axis=1)

# Binarize the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Reentrancy Vulnerability'])

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape X for CNN input (number of samples, number of features, 1)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))  # Change this to match the number of classes (vulnerability or not)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

#print('Accuracy:', accuracy_score(y_test_classes, y_pred_classes))
print('---------- CNN ----------')
print('Accuracy:', accuracy_score(y_test_classes, y_pred_classes))
print('Precision:', precision_score(y_test_classes, y_pred_classes))
print('Recall:', recall_score(y_test_classes, y_pred_classes))
print('F1 Score:', f1_score(y_test_classes, y_pred_classes))
print('----------------------------------------------')
print('Classification Report:\n', classification_report(y_test_classes, y_pred_classes))
print('Confusion Matrix:\n', confusion_matrix(y_test_classes, y_pred_classes))
