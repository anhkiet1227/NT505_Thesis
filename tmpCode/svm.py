import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import time

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the SVM model
#model = SVC(random_state=42, kernel='linear', gamma='auto')
#model = SVC(random_state=42, kernel='rbf', gamma='auto')
#model = SVC(random_state=42, kernel='poly', gamma='auto')
model = SVC(random_state=42, kernel='sigmoid', gamma='auto')

start_time = time.time()
# Train the model
model.fit(X_train, y_train)
end_time = time.time()
train_time = end_time - start_time

start_time = time.time()
# Predict on the test set
y_pred = model.predict(X_test)
end_time = time.time()
test_time = end_time - start_time

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('----------------------------------------------')
print('Training Time:', train_time)
print('Testing Time:', test_time)