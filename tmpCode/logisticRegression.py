import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import time

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

time_start = time.time()
# Train a Logistic Regression classifier for each vulnerability
lr_classifiers = {}
for vulnerability in y.columns:
    lr_classifier = LogisticRegression(random_state=42)
    lr_classifier.fit(X_train, y_train[vulnerability])
    lr_classifiers[vulnerability] = lr_classifier
time_end = time.time()
train_time = time_end - time_start
train_time = train_time

time_start = time.time()
# Predict on the test set for each vulnerability
y_preds = {vulnerability: classifier.predict(X_test) for vulnerability, classifier in lr_classifiers.items()}
time_end = time.time()
test_time = time_end - time_start
test_time = test_time

# Evaluate the classifiers for each vulnerability
for vulnerability in y.columns:
    print('----------', vulnerability, '----------')
    print('Accuracy:', accuracy_score(y_test[vulnerability], y_preds[vulnerability]))
    print('Precision:', precision_score(y_test[vulnerability], y_preds[vulnerability]))
    print('Recall:', recall_score(y_test[vulnerability], y_preds[vulnerability]))
    print('F1 Score:', f1_score(y_test[vulnerability], y_preds[vulnerability]))
    print('Confusion Matrix:\n', multilabel_confusion_matrix(y_test[vulnerability], y_preds[vulnerability]))


# You can also generate a detailed classification report
for vulnerability in y.columns:
    print('----------', vulnerability, '----------')
    print('Classification Report:\n', classification_report(y_test[vulnerability], y_preds[vulnerability]))

print('----------------------------------------------')
print('Training Time:', train_time)
print('Testing Time:', test_time)