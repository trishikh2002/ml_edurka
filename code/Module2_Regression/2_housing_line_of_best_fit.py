import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Housing-new.csv")

area = df["area"]
price_millions = df["price"] / 1000000

plt.figure()
plt.scatter(area, price_millions)

# Create x values for lines
x = np.linspace(area.min(), area.max(), 100)

# Improved possible regression lines (closer to data)
y1 = 0.00045 * x + 1.2
y2 = 0.00050 * x + 1.0
y3 = 0.00055 * x + 0.8
y4 = 0.00060 * x + 0.6
y5 = 0.00065 * x + 0.4

plt.plot(x, y1, label="Line 1")
plt.plot(x, y2, label="Line 2")
plt.plot(x, y3, label="Line 3")
plt.plot(x, y4, label="Line 4")
plt.plot(x, y5, label="Line 5")

plt.xlabel("Area (sq ft)")
plt.ylabel("Price (in millions)")
plt.title("Area vs Price with Possible Regression Lines")
plt.legend()
plt.show()
