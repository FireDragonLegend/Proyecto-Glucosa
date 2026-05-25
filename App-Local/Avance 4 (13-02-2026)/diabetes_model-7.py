"""
Módulo simple de Regresión Logística implementado en Python puro
para uso educativo en el proyecto: entrena un modelo sintético
y proporciona funciones para predecir la probabilidad de
desarrollar diabetes y generar un pequeño reporte.

No requiere dependencias externas.
"""
import math
import random


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


class LogisticRegressionModel:
    def __init__(self, n_features):
        # pesos incluyendo bias en la posición 0
        self.n = n_features
        self.w = [0.0] * (self.n + 1)

    def predict_proba(self, X):
        # X: lista de feature lists
        probs = []
        for x in X:
            z = self.w[0]
            for i, xi in enumerate(x):
                z += self.w[i+1] * xi
            probs.append(sigmoid(z))
        return probs

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def fit(self, X, y, lr=0.01, epochs=2000):
        n_samples = len(X)
        for epoch in range(epochs):
            # calcular predicciones
            preds = self.predict_proba(X)
            # gradiente para bias
            grad0 = 0.0
            grads = [0.0] * self.n
            for i in range(n_samples):
                error = preds[i] - y[i]
                grad0 += error
                for j in range(self.n):
                    grads[j] += error * X[i][j]
            # actualizar pesos
            self.w[0] -= lr * (grad0 / n_samples)
            for j in range(self.n):
                self.w[j+1] -= lr * (grads[j] / n_samples)


def generate_synthetic_dataset(n_samples=1000, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        # features: age, avg_fast, avg_post, a1c, care_score
        age = random.randint(20, 80)
        avg_fast = random.gauss(100, 30)
        avg_post = random.gauss(140, 40)
        a1c = random.gauss(5.6, 1.0)
        # care_score: 0 (excellent) .. 2 (poor)
        care = max(0.0, min(2.0, random.gauss(0.8, 0.8)))

        # score lineal (verdadero) que determina probabilidad
        score = (
            0.02 * (age - 40)
            + 0.03 * (avg_fast - 90)
            + 0.02 * (avg_post - 120)
            + 0.8 * (a1c - 5.0)
            + 1.2 * care
        )
        prob = sigmoid(score)
        label = 1 if random.random() < prob else 0
        X.append([age, avg_fast, avg_post, a1c, care])
        y.append(label)
    return X, y


def train_synthetic_model():
    X, y = generate_synthetic_dataset(1200)
    model = LogisticRegressionModel(n_features=5)
    model.fit(X, y, lr=0.0008, epochs=4000)
    return model


def map_probability_to_timeframe(prob):
    # prob in [0,1] -> estimate timeframe
    if prob >= 0.75:
        return "< 1 año"
    if prob >= 0.5:
        return "1-3 años"
    if prob >= 0.25:
        return "3-5 años"
    return "> 5 años"


def possible_symptoms_by_probability(prob):
    if prob >= 0.75:
        return [
            "Sed y hambre excesiva",
            "Orinar con frecuencia",
            "Fatiga constante",
            "Visión borrosa",
            "Cicatrización lenta"
        ]
    if prob >= 0.5:
        return [
            "Aumento de la sed",
            "Fatiga ocasional",
            "Aumento de la frecuencia urinaria"
        ]
    if prob >= 0.25:
        return [
            "Pequeñas variaciones de energía",
            "Hambre intermitente"
        ]
    return ["Bajo riesgo: pocos síntomas esperables en corto plazo"]
