import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def batch_data_loader(data_dir, batch_size=1000):
    """Yield mini-batches from multiple CSV files in data_dir."""
    files = glob.glob(os.path.join(data_dir, '**/*.csv'), recursive=True)
    for file in files:
        for chunk in pd.read_csv(file, chunksize=batch_size):
            yield chunk

def train_ensemble(data_dir, target_column, batch_size=1000):
    X_total, y_total = [], []
    for batch in batch_data_loader(data_dir, batch_size):
        X = batch.drop(columns=[target_column])
        y = batch[target_column]
        X_total.append(X)
        y_total.append(y)
    X_total = pd.concat(X_total)
    y_total = pd.concat(y_total)

    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.2, random_state=42)

    clf1 = RandomForestClassifier(n_estimators=50, random_state=42)
    clf2 = GradientBoostingClassifier(n_estimators=50, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2)], voting='soft')

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    print("Ensemble accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    data_dir = "path/to/your/data"
    target_column = "target"  # change to your target column name
    train_ensemble(data_dir, target_column, batch_size=1000)