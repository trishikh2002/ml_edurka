import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("email_spam_dataset.csv")
X = df[["Num_Emails"]]
y = df["Spam"]

# ----------------------------
# 2. Split into training and test sets
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# ----------------------------
# 3. Train Naive Bayes on training data
# ----------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# ----------------------------
# 4. Evaluate on test set
# ----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
report = classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam'])
print(report)

