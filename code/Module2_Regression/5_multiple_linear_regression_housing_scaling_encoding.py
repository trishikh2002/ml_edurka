import pandas as pd

# Step 1: Read dataset
df = pd.read_csv("Housing-new.csv")

# Separate features and target
X = df[["area", "bedrooms", "bathrooms", "furnishingstatus"]]
y = df["price"]

# Step 2: Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# Step 3: Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_features = ["area", "bedrooms", "bathrooms"]
categorical_features = ["furnishingstatus"]

# ColumnTranformer knows which columns are numeric, which columns are categorical, what to do with each
preprocessor = ColumnTransformer(
    transformers=[ # List of tuples, containing 3 parts: user-defined name, transformer, columns
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

# Apply preprocessing
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Step 4: Fit Multiple Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test_scaled)

# Print actual vs predicted side by side
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("--- Actual vs Predicted (Test Set) ---")
print(results)
print("--------------------------------------")

# Step 6: Visualization (Area vs Price for intuition)
import matplotlib.pyplot as plt

plt.scatter(X_test["area"], y_test)
plt.scatter(X_test["area"], y_pred)
plt.title("Price vs Area (Test set)")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.show()

# Step 7: New predictions
new_X_values_df = pd.DataFrame({
    "area": [1500, 2500, 4000, 6000, 8000],
    "bedrooms": [2, 2, 3, 3, 4],
    "bathrooms": [1, 2, 2, 3, 3],
    "furnishingstatus": [
        "unfurnished",
        "semi-furnished",
        "furnished",
        "furnished",
        "semi-furnished"
    ]
})

new_X_scaled = preprocessor.transform(new_X_values_df)
new_y_pred = model.predict(new_X_scaled)

print("\nPredictions for new houses:\n", new_y_pred)

# Step 8: Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)
