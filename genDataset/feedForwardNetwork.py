import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# Load the dataset
dataset_file = 'synthetic_smart_contract_dataset.csv'
data = pd.read_csv(dataset_file)

# Features (X) and labels (y)
# Exclude 'Contract Name' and 'Smart Contract Code'
X = data.drop(['Contract Name', 'Smart Contract Code', 'Reentrancy Vulnerability', 'Overflow/Underflow Vulnerability', 'Unprotected Ether Withdrawal'], axis=1)
# Assume each vulnerability is a separate label (binary classification for each vulnerability)
y = data[['Reentrancy Vulnerability', 'Overflow/Underflow Vulnerability', 'Unprotected Ether Withdrawal']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Feedforward Neural Network for each vulnerability
mlp_classifiers = {}
for vulnerability in y.columns:
    mlp_classifier = MLPClassifier(random_state=42)
    mlp_classifier.fit(X_train, y_train[vulnerability])
    mlp_classifiers[vulnerability] = mlp_classifier

# Predict on the test set for each vulnerability
y_preds = {vulnerability: classifier.predict(X_test) for vulnerability, classifier in mlp_classifiers.items()}

# Evaluate the classifiers for each vulnerability
for vulnerability in y.columns:
    print('----------', vulnerability, '----------')
    print('Accuracy:', accuracy_score(y_test[vulnerability], y_preds[vulnerability]))
    print('Precision:', precision_score(y_test[vulnerability], y_preds[vulnerability]))
    print('Recall:', recall_score(y_test[vulnerability], y_preds[vulnerability]))
    print('F1 Score:', f1_score(y_test[vulnerability], y_preds[vulnerability]))
    print('Classification Report:\n', classification_report(y_test[vulnerability], y_preds[vulnerability]))
