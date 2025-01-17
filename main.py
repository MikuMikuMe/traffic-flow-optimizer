Creating a complete Python program for a project like "traffic-flow-optimizer" involves several steps, including data collection, preprocessing, model training, prediction, and user interaction. Below is a simplified version of such a program. This example assumes access to real-time traffic data, perhaps from a service like Open Traffic or a local Department of Transportation API.

For a real-time system, you'd typically separate the data collection and model training processes from the real-time prediction system. However, this example includes a single script for simplicity.

### Traffic Flow Optimizer

```python
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrafficFlowOptimizer:
    def __init__(self):
        # Initialize model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def fetch_data(self):
        # Fetch data from an API (mocked here)
        logging.info("Fetching traffic data...")
        try:
            # In practice, replace this with a call to real traffic API
            traffic_data = {
                'time': np.arange(0, 1000),
                'traffic_volume': np.random.randint(50, 250, size=1000),  # Simulated data
                'weather_conditions': np.random.choice(['sunny', 'rainy', 'cloudy'], size=1000)
            }
            df = pd.DataFrame(traffic_data)
            logging.info("Data fetched successfully.")
            return df
        except Exception as e:
            logging.error(f"Failed to fetch data: {e}")
            return pd.DataFrame()

    def preprocess_data(self, df):
        # Preprocess the data
        logging.info("Preprocessing data...")
        try:
            df['weather_conditions'] = df['weather_conditions'].map({'sunny': 0, 'rainy': 1, 'cloudy': 2})
            df = df.dropna()

            X = df[['time', 'weather_conditions']]
            y = df['traffic_volume']
            X_scaled = self.scaler.fit_transform(X)
            
            logging.info("Data preprocessing completed.")
            return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            return None, None, None, None

    def train_model(self, X_train, y_train):
        # Train the model
        logging.info("Training model...")
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed.")
        except Exception as e:
            logging.error(f"Model training failed: {e}")

    def predict_traffic(self, X_test):
        # Predict using the model
        logging.info("Predicting traffic volume...")
        try:
            predictions = self.model.predict(X_test)
            logging.info("Prediction completed.")
            return predictions
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None

    def run(self):
        # Run the optimizer
        logging.info("Starting Traffic Flow Optimizer...")
        df = self.fetch_data()
        if df.empty:
            logging.error("No data available, exiting...")
            return

        X_train, X_test, y_train, y_test = self.preprocess_data(df)
        if X_train is None:
            logging.error("Preprocessing failed, exiting...")
            return

        self.train_model(X_train, y_train)
        predictions = self.predict_traffic(X_test)

        # Display results
        if predictions is not None:
            for i, prediction in enumerate(predictions[:5]):  # Display first 5 predictions
                print(f"Predicted traffic volume: {prediction:.2f} at time {i}")

if __name__ == "__main__":
    optimizer = TrafficFlowOptimizer()
    optimizer.run()
```

### Key Features and Considerations:

1. **Data Fetching:** Replace the `fetch_data` method with actual API calls to get real-time data. Handle possible exceptions, such as HTTP errors or connection issues.

2. **Data Preprocessing:** This step includes data cleaning, encoding categorical variables, and scaling features. Error handling ensures that issues in preprocessing do not crash the program.

3. **Model Training:** The script uses a `RandomForestRegressor` for simplicity, but other models may be more suited based on the problem's complexity and size. Ensure to try different model configurations.

4. **Real-time Prediction:** Predictions are done on test data here for demonstration purposes. For real applications, integrate this with real-time prediction pipelines.

5. **Logging:** The use of Python's logging module helps track the program's execution and simplifies debugging.

6. **Error Handling:** Wrapping potential points of failure in try-except blocks helps gracefully handle and log errors, making maintenance easier.