import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessing():
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, path):
        data = pd.read_csv(path, sep="\t", header=None).dropna()

        print(data.head())

        data = data.to_numpy()

        return data
    
    def preprocess(self, data, test_size=0.2, validation_size=0.2, random_state=12):
        """Split data into train, validation, and test sets, and normalize features."""
        # Split into train-validation and test sets
        train_validation, test = train_test_split(data, test_size=test_size, random_state=random_state)
        
        # Further split into train and validation sets
        train, validation = train_test_split(train_validation, test_size=validation_size, random_state=random_state)

        # Fit and apply scaling
        self.scaler.fit(train[:, :-1])
        X_train = self.scaler.transform(train[:, :-1])
        X_val = self.scaler.transform(validation[:, :-1])
        X_test = self.scaler.transform(test[:, :-1])

        # Extract labels
        y_train = train[:, -1]
        y_val = validation[:, -1]
        y_test = test[:, -1]

        return X_train, y_train, X_val, y_val, X_test, y_test