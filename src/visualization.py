import matplotlib
matplotlib.use('TkAgg')  # Forcer l'utilisation de TkAgg

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from src.data_processing import load_data, prepare_data
from src.model import IrisClassifier
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, title='Matrice de Confusion'):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()

def plot_decision_boundaries(classifier, X, y):
    # Sélectionner deux features pour la visualisation 2D
    feature1, feature2 = 0, 1  # Première et deuxième caractéristiques
    
    # Créer une grille de points
    x_min, x_max = X.iloc[:, feature1].min() - 1, X.iloc[:, feature1].max() + 1
    y_min, y_max = X.iloc[:, feature2].min() - 1, X.iloc[:, feature2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # Faire des prédictions pour chaque point de la grille
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel(), 
                               np.zeros_like(xx.ravel()), 
                               np.zeros_like(xx.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Tracer les frontières de décision
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X.iloc[:, feature1], X.iloc[:, feature2], c=y, alpha=0.8)
    plt.xlabel(X.columns[feature1])
    plt.ylabel(X.columns[feature2])
    plt.title('Frontières de Décision (2 premières caractéristiques)')
    plt.tight_layout()

def plot_error_analysis(X_test, y_test, y_pred):
    # Identifier les erreurs
    errors = y_test != y_pred
    X_errors = X_test[errors]
    y_true_errors = y_test[errors]
    y_pred_errors = y_pred[errors]
    
    # Visualiser les erreurs
    plt.figure(figsize=(10, 6))
    for feature in X_test.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test[feature], y_test, label='Correct', alpha=0.5)
        plt.scatter(X_errors[feature], y_true_errors, 
                   label='Erreur', color='red', alpha=0.7)
        plt.xlabel(feature)
        plt.ylabel('Classe')
        plt.title(f'Analyse des erreurs - {feature}')
        plt.legend()
        plt.tight_layout()

def create_visualizations(X, y):
    # Combine features and target for easier plotting
    data = X.copy()
    data['species'] = y
    
    # 1. Pairplot
    print("Création du pairplot...")
    sns.pairplot(data, hue='species')
    plt.tight_layout()
    
    # 2. Feature distributions
    print("Création des distributions...")
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(X.columns):
        plt.subplot(2, 2, i+1)
        for species in y.unique():
            subset = data[data['species'] == species]
            sns.kdeplot(data=subset[feature], label=f'Classe {species}')
        plt.title(f'Distribution de {feature}')
        plt.xlabel('Valeur')
        plt.ylabel('Densité')
    plt.tight_layout()

def main():
    # 1. Charger et préparer les données
    print("Chargement des données...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # 2. Créer et entraîner le modèle
    print("\nEntraînement du modèle...")
    classifier = IrisClassifier()
    classifier.train(X_train, y_train)
    
    # 3. Évaluer le modèle
    print("\nÉvaluation du modèle:")
    accuracy, report = classifier.evaluate(X_test, y_test)
    print(f"\nPrécision: {accuracy:.2f}")
    print("\nRapport détaillé:")
    print(report)
    
    # 4. Faire des prédictions
    y_pred = classifier.predict(X_test)
    
    # 5. Créer toutes les visualisations
    print("\nCréation des visualisations...")
    
    # Visualisations de base
    create_visualizations(X, y)
    
    # Matrice de confusion
    print("Création de la matrice de confusion...")
    plot_confusion_matrix(y_test, y_pred)
    
    # Frontières de décision
    print("Création des frontières de décision...")
    plot_decision_boundaries(classifier, X, y)
    
    # Analyse des erreurs
    print("Analyse des erreurs...")
    plot_error_analysis(X_test, y_test, y_pred)
    
    # 6. Afficher toutes les visualisations
    print("\nAffichage des visualisations...")
    plt.show()

if __name__ == "__main__":
    main()