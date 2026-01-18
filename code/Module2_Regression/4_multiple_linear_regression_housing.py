import pandas as pd

# Step 1: Read dataset
df = pd.read_csv("Housing-new.csv")

# Multiple independent variables
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"] / 1000000   # scale price to millions

# Step 2: Split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# Print X and y for the test set
print("--- Test Set Data ---")
print("X_test:\n", X_test)
print("\ny_test (Price in millions):\n", y_test)
print("---------------------\n")

# Step 3: Fit Multiple Linear Regression to Training Data
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make Prediction
y_pred = model.predict(X_test)

# Print actual vs predicted side by side
results = pd.DataFrame({
    "Area (sq ft)": X_test["area"].values,
    "Bedrooms": X_test["bedrooms"].values,
    "Bathrooms": X_test["bathrooms"].values,
    "Actual Price (millions)": y_test.values,
    "Predicted Price (millions)": y_pred
})

print("--- Actual vs Predicted (Test Set) ---")
print(results)
print("--------------------------------------")

# Step 5: Visualize training set results (Area vs Price)
import matplotlib.pyplot as plt

plt.scatter(X_train["area"], y_train)
plt.scatter(X_train["area"], model.predict(X_train))
plt.title("Price vs Area (Training set)")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (in millions)")
plt.show()

# Step 6: Visualize test set results (Area vs Price)
plt.scatter(X_test["area"], y_test)
plt.scatter(X_test["area"], y_pred)
plt.title("Price vs Area (Test set)")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (in millions)")
plt.show()

# Step 7: Make new predictions
new_X_values_df = pd.DataFrame({
    "area": [1500, 2500, 4000, 6000, 8000],
    "bedrooms": [2, 2, 3, 3, 4],
    "bathrooms": [1, 2, 2, 3, 3]
})

new_y_pred = model.predict(new_X_values_df)
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
