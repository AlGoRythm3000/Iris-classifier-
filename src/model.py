from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class IrisClassifier:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
    
    def train(self, X_train, y_train):
        """Entraîne le modèle"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Fait des prédictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Évalue le modèle"""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report