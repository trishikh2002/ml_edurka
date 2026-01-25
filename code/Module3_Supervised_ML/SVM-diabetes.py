import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("diabetes.csv")

# ----------------------------
# 2. Feature and target
# ----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ----------------------------
# 3. Train-Test Split (80-20)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ----------------------------
# 4. Feature Scaling (MANDATORY for SVM)
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 5. Train SVM model
# ----------------------------
model = SVC(
    kernel="rbf",     # non-linear boundary
    C=1.0,            # regularization
    gamma="scale",    # kernel width
    probability=True # enables predict_proba
)
model.fit(X_train_scaled, y_train)

# ----------------------------
# 6. Predict labels on TEST set
# ----------------------------
y_pred = model.predict(X_test_scaled)

# ----------------------------
# 7. Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes (0)", "Diabetes (1)"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (SVM)")
plt.show()

# ----------------------------
# 8. Evaluation Metrics
# ----------------------------
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("\n=== Test Set Performance ===")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1 Score : {f1:.3f}")

# -------------------------------
# 9. New (unseen) patient record
# -------------------------------
new_patient = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 0,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 35
}])

print("\nNew (Unseen) Patient Record:")
print(new_patient)

new_patient_scaled = scaler.transform(new_patient)

prediction = model.predict(new_patient_scaled)[0]
probability = model.predict_proba(new_patient_scaled)[0][1]

print("Predicted Outcome:", prediction)
print(f"Predicted Probability of Diabetes: {probability:.3f}")
