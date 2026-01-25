import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
y = df["Outcome"]   # used ONLY for evaluation, not training

# ----------------------------
# 3. Train-Test Split (80-20)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ----------------------------
# 4. Feature Scaling (MANDATORY for K-Means)
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 5. Train K-Means model
# ----------------------------
kmeans = KMeans(
    n_clusters=2,        # diabetes / no diabetes - k-means does not know what is diabetes, just wants 2 groups
    random_state=42,
    n_init=10 # Run k-means 10 times with different random centers, finally choosing the best one
)
kmeans.fit(X_train_scaled)

# ----------------------------
# 6. Predict cluster labels on TEST set
# ----------------------------
y_cluster = kmeans.predict(X_test_scaled)

# ----------------------------
# 7. Align clusters with actual labels: If most diabetic patients are not in cluster 1, flip the cluster labels
# ----------------------------
# K-Means cluster labels (0 or 1) are arbitrary and may not correspond to:
# 0 = no diabetes, 1 = diabetes (which our confusion matrix assumes)
# So we align cluster labels with actual Outcome labels before evaluation
#
# y_test == 1 selects only actual diabetic patients
# y_cluster[y_test == 1] gives cluster labels assigned to those diabetic patients
#
# Example:
# y_cluster[y_test == 1] = [0, 0, 1, 0, 0]
# This means: among 5 truly diabetic patients,
# - 4 were assigned to cluster 0
# - 1 was assigned to cluster 1
#
# np.mean(...) computes the fraction of diabetic patients in cluster 1
# Here: mean = 0.2
#
# If less than 50% of diabetic patients are in cluster 1,
# then cluster 1 is likely NOT the diabetes cluster
# So we flip the labels so that:
# 0 = no diabetes, 1 = diabetes
#
# y_cluster = 1 - y_cluster flips labels: 0 → 1, 1 → 0
if np.mean(y_cluster[y_test == 1]) < 0.5:
    y_cluster = 1 - y_cluster

# ----------------------------
# 8. Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_cluster)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes (0)", "Diabetes (1)"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (K-Means)")
plt.show()

# ----------------------------
# 9. New (unseen) patient record
# ----------------------------
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

cluster = kmeans.predict(new_patient_scaled)[0]

print("Assigned Cluster:", cluster)
