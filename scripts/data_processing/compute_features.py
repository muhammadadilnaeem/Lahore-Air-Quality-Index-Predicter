import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
data = pd.read_csv(r"E:\Project and Advices\Project Air Quality Index project\data\raw\lahore_air_quality_data.csv")

# Feature engineering
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour
data['weekday'] = data['date'].dt.weekday

# Drop columns that won't be used
data = data.drop(columns=['locationId', 'location', 'parameter', 'unit', 'coordinates', 'country', 'city', 'isMobile', 'isAnalysis', 'entity', 'sensorType'])

# Define features and target variable
X = data[['month', 'day', 'hour', 'weekday']]
y = data['value']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
processed_data_path = "E:/Project and Advices/Project Air Quality Index project/data/processed/"
X_train.to_csv(processed_data_path + 'X_train.csv', index=False)
X_test.to_csv(processed_data_path + 'X_test.csv', index=False)
y_train.to_csv(processed_data_path + 'y_train.csv', index=False)
y_test.to_csv(processed_data_path + 'y_test.csv', index=False)
