import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

# Load the dataset
dataset_file = 'synthetic_smart_contract_dataset.csv'
data = pd.read_csv(dataset_file)

# Prepare the data
X = data['Smart Contract Code']  # Smart contract code
y = data['Reentrancy Vulnerability']  # Vulnerability: 1 for present, 0 for absent

# Load the pre-trained SBERT model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Encode contract code into embeddings using SBERT
X_embeddings = model.encode(X, convert_to_tensor=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

# Train a Logistic Regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
