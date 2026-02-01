# ----------------------------
# Step 1: Load Data
# ----------------------------
import pandas as pd

dataset = pd.read_csv('salary.csv')

# Features and target as DataFrames
X = dataset[['YearsExperience']]  # shape (n_samples, 1)
y = dataset[['Salary']]

print("Feature (X) head:\n", X.head())
print("Target (y) head:\n", y.head())

# ----------------------------
# Step 2: Split data into training and testing
# ----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0
)

# ----------------------------
# Step 3: Fit Simple Linear Regression
# ----------------------------
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ----------------------------
# Step 4: Make Predictions
# ----------------------------
y_pred = regressor.predict(X_test)
print("\nPredictions on Test Set:\n", y_pred)

# ----------------------------
# Step 5: Visualize Training Set
# ----------------------------
import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color='red', label='Actual')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Regression Line')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# ----------------------------
# Step 6: Visualize Test Set
# ----------------------------
plt.scatter(X_test, y_test, color='red', label='Actual')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Regression Line')  # line is same as training
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# ----------------------------
# Step 7: Make new predictions (as DataFrame)
# ----------------------------
import joblib

# Save column names for future use
joblib.dump(list(X.columns), 'column_names.pkl')

# Save trained model
joblib.dump(regressor, 'final_model.pkl')

# Load model and column names
loaded_model = joblib.load('final_model.pkl')
col_names = joblib.load('column_names.pkl')

# Make predictions safely using DataFrame with correct columns
new_experience_years = [5, 10, 15, 20]
new_data = pd.DataFrame(new_experience_years, columns=col_names)
new_salary_pred = loaded_model.predict(new_data)

for exp, salary in zip(new_experience_years, new_salary_pred):
    print(f"Predicted salary for {exp} years experience: {salary[0]:.2f}")
