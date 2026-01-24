import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("email_spam_dataset.csv")

# ----------------------------
# 2. Feature and target
# ----------------------------
X = df[["Num_Emails"]]
y = df["Spam"]

# ----------------------------
# 3. Train-Test Split (80-20)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ----------------------------
# 4. Train logistic regression
# ----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ----------------------------
# 5. Predict probabilities and labels on TEST set
# ----------------------------
y_prob = model.predict_proba(X_test)[:, 1]   # Probability that Spam = 1
y_pred = (y_prob >= 0.5).astype(int)         # Decision boundary = 0.5

# ----------------------------
# 6. Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Spam (0)", "Spam (1)"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.show()

# ----------------------------
# 7. Evaluation Metrics
# ----------------------------
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print(f"\n=== Test Set Performance ===")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1 Score : {f1:.3f}")

# ----------------------------
# 8. Logistic Regression Curve
# ----------------------------
# Create smooth curve for visualization
# creates 200 evenly spaced values between the smallest and largest Num_Emails in our dataset
# X.min() → the smallest number of emails seen
# X.max() → the largest number of emails seen
# 200 → means “generate 200 points in between”, which will also make it smooth
# reshape() needed, because linspace gives 1D array, but sklearn expects row x column format, so convert array such as [0, 1, 2, 3, ...] into 
#[
# [0], 
# [1.01], 
# [2.02], 
# [3.03], 
# ...]

# Create 200 equally spaced points on the X-axis for plotting
X_curve = pd.DataFrame(
    np.linspace(X.min().values[0], X.max().values[0], 200),
    columns=["Num_Emails"]
)


# For each of the 200 values in x_test, predict the probability of being spam
y_curve = model.predict_proba(X_curve)[:, 1]

plt.figure(figsize=(7,5))
# Plot training data
plt.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.5)
# Plot test data
plt.scatter(X_test, y_test, color="red", label="Test Data", alpha=0.7, marker='x', s=100)
# Plot logistic curve
plt.plot(X_curve, y_curve, color="green", linewidth=2, label="Logistic Regression Curve")
plt.title("Logistic Regression: Num_Emails vs Spam Probability")
plt.xlabel("Number of Emails")
plt.ylabel("Predicted Probability of Spam")
plt.legend()
plt.show()