import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("Heart.csv")

# ----------------------------
# 2. Feature and target
# ----------------------------
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# ----------------------------
# 3. Identify categorical and numerical columns
# ----------------------------
categorical_cols = [
    "Gender",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

numerical_cols = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak"
]

# ----------------------------
# 4. Column Transformer with OHE + Scaling
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

# ----------------------------
# 5. Train-Test Split (80-20)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ----------------------------
# 6. Apply preprocessing
# ----------------------------
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# ----------------------------
# 7. Train logistic regression
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_encoded, y_train)

# ----------------------------
# 8. Predict probabilities and labels on TEST set
# ----------------------------
y_prob = model.predict_proba(X_test_encoded)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# ----------------------------
# 9. Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease (0)", "Disease (1)"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.show()

# ----------------------------
# 10. Evaluation Metrics
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
