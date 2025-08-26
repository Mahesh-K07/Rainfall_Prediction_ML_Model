import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load your dataset
df = pd.read_csv('rainfall_data.csv')  # Sample dataset should have features + 'Rainfall'

# Features and target
X = df.drop('Rainfall', axis=1)
y = df['Rainfall']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('rainfall_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
