# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic
import pickle

# Load dataset
df = pd.read_csv('amazon_delivery.csv')

# Drop rows with missing values
df = df.dropna()

# Feature Engineering
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Order_Time'] = pd.to_datetime(df['Order_Time']).dt.hour
df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time']).dt.hour

# Extract useful date information
df['Day_of_Week'] = df['Order_Date'].dt.dayofweek
df['Month'] = df['Order_Date'].dt.month

# Calculate distance between store and drop-off
df['Distance'] = df.apply(lambda row: geodesic((row['Store_Latitude'], row['Store_Longitude']),
                                               (row['Drop_Latitude'], row['Drop_Longitude'])).km, axis=1)

# Encode categorical columns
label_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Select features and target
features = ['Agent_Age', 'Agent_Rating', 'Order_Time', 'Pickup_Time', 'Day_of_Week', 'Month', 'Distance', 'Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
X = df[features]
y = df['Delivery_Time']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Root Mean Squared Error: {rmse}')

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully!")



# Import necessary libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# After predicting and evaluating, plot actual vs predicted values
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)  # Line showing perfect prediction
plt.xlabel('Actual Delivery Time')
plt.ylabel('Predicted Delivery Time')
plt.title('Actual vs Predicted Delivery Time')
plt.show()

# If you prefer a residual plot
sns.residplot(x=y_test, y=y_pred, lowess=True, color="g")
plt.xlabel('Actual Delivery Time')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.show()
