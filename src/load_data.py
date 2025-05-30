import os
import pandas as pd

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "data", "raw"))
    print("Looking for data in:", data_dir)

    train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    oil = pd.read_csv(os.path.join(data_dir, "oil.csv"), parse_dates=["date"])
    holidays = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"), parse_dates=["date"])
    transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"), parse_dates=["date"])

    return train, test, stores, oil, holidays, transactions

if __name__ == "__main__":
    load_data()  # Now you can run `uv run .\src\load_data.py`
