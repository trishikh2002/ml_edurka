import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------------
# 4. Train Decision Tree (Entropy)
# ----------------------------
model = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=4)
model.fit(X_train, y_train)

# ----------------------------
# 5. Predictions & Evaluation
# ----------------------------
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(cm, display_labels=["Not Spam (0)", "Spam (1)"]).plot(cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# 6. Visualize the Tree
# ----------------------------
plt.figure(figsize=(14, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Spam", "Spam"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree using Entropy")
plt.show()

# ----------------------------
# 7. Print the actual decision rules
# ----------------------------
tree_rules = export_text(model, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n")
print(tree_rules)

# ----------------------------
# 8. Feature Importance (Entropy-based)
# ----------------------------
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (based on Information Gain):\n")
print(importances)

plt.figure(figsize=(8, 5))
plt.barh(importances["Feature"], importances["Importance"], color="teal")
plt.gca().invert_yaxis()
plt.title("Feature Importance (Entropy-Based)")
plt.xlabel("Information Gain Contribution")
plt.show()
