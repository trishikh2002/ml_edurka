import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Housing-new.csv")

area = df["area"]
price_millions = df["price"] / 1000000  # convert to millions

plt.figure()
plt.scatter(area, price_millions)
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (in millions)")
plt.title("Area vs Price (Scaled)")
plt.show()
