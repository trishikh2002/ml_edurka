import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("email_spam_dataset.csv")

# ----------------------------
# 2. Define features and target
# ----------------------------
X = df.drop("Spam", axis=1)
y = df["Spam"]

# ----------------------------
# 3. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. Train Bagging Classifier (Decision Trees)
# ----------------------------
base_tree = DecisionTreeClassifier(
    criterion="entropy",  # optional, shows info gain
    max_depth=6,
    random_state=42
)

bagging_model = BaggingClassifier(
    estimator=base_tree,
    n_estimators=100,
    random_state=42,
    bootstrap=True
)

bagging_model.fit(X_train, y_train)

# ----------------------------
# 5. Predictions & Evaluation
# ----------------------------
y_pred = bagging_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    cm, display_labels=["Not Spam (0)", "Spam (1)"]
).plot(cmap="Blues")

plt.title("Bagging Classifier Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# 6. Feature Importance (average across trees)
# ----------------------------
# BaggingClassifier does not have feature_importances_ directly,
# so we can compute the average from all trees manually
all_importances = np.array([tree.feature_importances_ for tree in bagging_model.estimators_])
avg_importances = np.mean(all_importances, axis=0)

importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": avg_importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (Bagging - Average over Trees):\n")
print(importances)

plt.figure(figsize=(8, 5))
plt.barh(importances["Feature"], importances["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Bagging Classifier)")
plt.xlabel("Average Information Gain")
plt.show()

# ----------------------------
# 7. Inspect one tree from the ensemble (optional)
# ----------------------------
plt.figure(figsize=(14, 6))
plot_tree(
    bagging_model.estimators_[0],
    feature_names=X.columns,
    class_names=["Not Spam", "Spam"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("One Decision Tree from the Bagging Ensemble")
plt.show()
