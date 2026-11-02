
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from pathlib import Path

# Create models folder if it does not exist
Path("models").mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv("data/student_data.csv")

# Separate input features and target
X = df.drop("final_score", axis=1)
y = df["final_score"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Save trained model
joblib.dump(model, "models/student_model.pkl")

print("Model trained and saved successfully!")
