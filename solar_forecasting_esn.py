import sys
import subprocess
import os

# --- Clone and install simple_esn if not already installed ---
if not os.path.exists('simple_esn'):
    subprocess.check_call(['git', 'clone', 'https://github.com/sylvchev/simple_esn.git'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', './simple_esn'])

from simple_esn.simple_esn import SimpleESN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(file_path):
    """Load dataset from Excel file."""
    try:
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        print("❌ The file was not found.")
        return None

def preprocess_data(df):
    """Preprocess data by stripping spaces from column names and renaming."""
    df.columns = df.columns.str.strip()
    df.rename(columns={"Sloar Power": "Solar Power"}, inplace=True, errors='ignore')
    return df

def inspect_data(df):
    """Inspect data types and view the first few rows."""
    print("📊 Data Types:")
    print(df.dtypes)
    print("📝 First Few Rows:")
    print(df.head())

def prepare_features(df):
    """Prepare required features and target."""
    features = ["Temperature Units", "Pressure Units", "Relative Humidity Units", "Wind Speed Units"]
    target = "Solar Power"
    required_columns = features + [target]

    # Attempt to convert columns to numeric
    for col in required_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values
    df = df.dropna(subset=required_columns)

    # Check if any columns were excluded due to non-numeric data
    excluded_columns = [col for col in required_columns if df[col].dtype not in ['int64', 'float64']]
    if excluded_columns:
        print("❌ Excluded columns due to non-numeric data:", excluded_columns)
        return None, None

    print("✅ Dataframe successfully filtered with required columns!")

    X = df[features].values
    y = df[target].values
    return X, y

def train_esn_model(X, y):
    """Train an Echo State Network and calculate regression metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize ESN
    esn = SimpleESN(n_readout=100, n_components=200, random_state=42)

    # Fit on training data
    esn.fit(X_train, y_train)
    # Predict on test data
    y_pred = esn.transform(X_test)

    # Flatten y_pred if it's 2D
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"ESN Model MSE: {mse:.4f}")
    print(f"ESN Model MAE: {mae:.4f}")
    print(f"ESN Model RMSE: {rmse:.4f}")
    print(f"ESN Model R^2 (Coefficient of Determination): {r2:.4f}")

    return esn

def main():
    file_path = "Data_solar_on_27-04-2022.xlsx"
    df = load_data(file_path)

    if df is not None:
        df = preprocess_data(df)
        inspect_data(df)
        X, y = prepare_features(df)

        if X is not None and y is not None:
            model = train_esn_model(X, y)

if __name__ == "__main__":
    main()
