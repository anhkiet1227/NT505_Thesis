import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset_file = 'synthetic_smart_contract_dataset.csv'
data = pd.read_csv(dataset_file)

# Features (X) and labels (y)
X = data.drop(['Contract Name', 'Smart Contract Code', 'Reentrancy Vulnerability', 'Overflow/Underflow Vulnerability', 'Unprotected Ether Withdrawal'], axis=1)

# Binarize the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Reentrancy Vulnerability'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the XGBoost model
model = XGBClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print('---------- XG Boost ----------')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('----------------------------------------------')
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))