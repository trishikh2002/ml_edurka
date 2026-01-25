import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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
# 4. Feature Scaling (VERY important for KNN)
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 5. Train KNN model
# ----------------------------
k = 5
model = KNeighborsClassifier(n_neighbors=k)
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
plt.title(f"Confusion Matrix (KNN, k={k})")
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

new_patient_scaled = scaler.transform(new_patient)

print("\nNew (Unseen) Patient Record:")
print(new_patient)

prediction = model.predict(new_patient_scaled)[0]
print("Predicted Outcome:", prediction)

# Find k nearest neighbors for this new record
# Internally, this will do this: 
# Distance = Square root of ((Glucose1 - Glucose2) ^ 2 + (BMI1 - BMI2) ^ 2 + ...)
# Then for the nearest 5, voting will have decided earlier if diabetic or not
distances, indices = model.kneighbors(new_patient_scaled)

neighbors = X_train.iloc[indices[0]].copy()
neighbors["Actual Outcome"] = y_train.iloc[indices[0]].values
neighbors["Distance"] = distances[0]

print("\nNearest Neighbors:")
print(neighbors)
