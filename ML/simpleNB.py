import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SimpleNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = {}
        self.feature_counts = {}
        
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = X_cls.shape[0] / X.shape[0]
            self.feature_counts[cls] = np.sum(X_cls, axis=0)

        self.total_feature_counts = np.sum(list(self.feature_counts.values()), axis=0)

    def predict(self, X):
        predictions = []
        
        for x in X:
            class_probabilities = {}
            
            for cls in self.classes:
                likelihood = self.feature_counts[cls] / self.total_feature_counts
                posterior = self.class_priors[cls] * np.prod(likelihood ** x)
                class_probabilities[cls] = posterior

            predictions.append(max(class_probabilities, key=class_probabilities.get))
        
        return np.array(predictions)

if __name__ == "__main__":
    data = {
        'Feature1': [1, 1, 0, 0, 1, 0],
        'Feature2': [0, 1, 1, 0, 1, 0],
        'Label': ['A', 'A', 'B', 'B', 'A', 'B']
    }
    df = pd.DataFrame(data)
    X = df[['Feature1', 'Feature2']].values
    y = df['Label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = SimpleNaiveBayes()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
