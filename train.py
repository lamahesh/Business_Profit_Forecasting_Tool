import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("cs5_jan26.csv")

# Check for null values
print("Null values:\n", data.isnull().sum())

# Features and target
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
y = data['Profit']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("profit_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully!")
