import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Create folders if not exist
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------------
# 1. CREATE SYNTHETIC DATASET
# -----------------------------
np.random.seed(42)

data = pd.DataFrame({
    'area': np.random.randint(500, 3000, 200),
    'bedrooms': np.random.randint(1, 5, 200),
    'bathrooms': np.random.randint(1, 4, 200),
    'age': np.random.randint(0, 20, 200),
    'parking': np.random.randint(0, 2, 200),
})

data['price'] = (
    data['area'] * 3000 +
    data['bedrooms'] * 500000 +
    data['bathrooms'] * 300000 -
    data['age'] * 10000 +
    data['parking'] * 200000 +
    np.random.randint(-100000, 100000, 200)
)

print("Dataset Preview:\n", data.head())

# -----------------------------
# 2. EDA
# -----------------------------
sns.pairplot(data)
plt.savefig("images/pairplot.png")
plt.close()

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("images/heatmap.png")
plt.close()

# -----------------------------
# 3. SPLIT DATA
# -----------------------------
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. MODELS
# -----------------------------
lr = LinearRegression()
rf = RandomForestRegressor()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -----------------------------
# 5. EVALUATION FUNCTION
# -----------------------------
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    return mae, rmse, r2, pred

lr_metrics = evaluate(lr, X_test, y_test)
rf_metrics = evaluate(rf, X_test, y_test)

print("\nLinear Regression Metrics:")
print("MAE:", lr_metrics[0])
print("RMSE:", lr_metrics[1])
print("R2 Score:", lr_metrics[2])

print("\nRandom Forest Metrics:")
print("MAE:", rf_metrics[0])
print("RMSE:", rf_metrics[1])
print("R2 Score:", rf_metrics[2])

# -----------------------------
# 6. SAVE MODEL
# -----------------------------
joblib.dump(rf, 'models/house_model.pkl')

# -----------------------------
# 7. VISUALIZATION
# -----------------------------
plt.scatter(y_test, rf_metrics[3])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.savefig("images/prediction.png")
plt.close()

# -----------------------------
# 8. SAMPLE PREDICTION
# -----------------------------
sample = np.array([[1500, 3, 2, 5, 1]])
prediction = rf.predict(sample)

print("\nSample House Prediction:")
print("Input: Area=1500, Bedrooms=3, Bathrooms=2, Age=5, Parking=1")
print("Predicted Price:", int(prediction[0]))