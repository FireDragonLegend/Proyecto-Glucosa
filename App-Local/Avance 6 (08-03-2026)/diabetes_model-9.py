"""
Módulo de Red Neuronal Avanzada para predicción de diabetes.
Incluye evaluación de prediabetes, prediabetes infantil y múltiples factores de riesgo
con mayor precisión y coherencia médica usando redes neuronales profundas.
"""
import math
import random
import numpy as np


def sigmoid(z):
    """Función sigmoide con manejo de overflow."""
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 1.0 if z > 0 else 0.0


def relu(z):
    """Función ReLU."""
    return max(0.0, z)


def relu_derivative(z):
    """Derivada de ReLU."""
    return 1.0 if z > 0 else 0.0


def tanh(z):
    """Función tangente hiperbólica."""
    try:
        return math.tanh(z)
    except OverflowError:
        return 1.0 if z > 0 else -1.0


def tanh_derivative(z):
    """Derivada de tanh."""
    t = tanh(z)
    return 1.0 - t * t


class AdvancedNeuralNetwork:
    """
    Red Neuronal Avanzada para predicción de diabetes.
    Arquitectura: Input -> Hidden1 (ReLU) -> Hidden2 (ReLU) -> Output (Sigmoid)
    Incluye regularización L2, dropout y optimización Adam.
    """

    def __init__(self, input_size, hidden1_size=64, hidden2_size=32, output_size=1,
                 learning_rate=0.001, l2_lambda=0.001, dropout_rate=0.2):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

        # Inicialización de pesos con Xavier/Glorot
        self.W1 = self._initialize_weights(input_size, hidden1_size)
        self.b1 = [0.0] * hidden1_size
        self.W2 = self._initialize_weights(hidden1_size, hidden2_size)
        self.b2 = [0.0] * hidden2_size
        self.W3 = self._initialize_weights(hidden2_size, output_size)
        self.b3 = [0.0] * output_size

        # Parámetros de Adam
        self.m_W1 = [[0.0] * hidden1_size for _ in range(input_size)]
        self.v_W1 = [[0.0] * hidden1_size for _ in range(input_size)]
        self.m_b1 = [0.0] * hidden1_size
        self.v_b1 = [0.0] * hidden1_size

        self.m_W2 = [[0.0] * hidden2_size for _ in range(hidden1_size)]
        self.v_W2 = [[0.0] * hidden2_size for _ in range(hidden1_size)]
        self.m_b2 = [0.0] * hidden2_size
        self.v_b2 = [0.0] * hidden2_size

        self.m_W3 = [[0.0] * output_size for _ in range(hidden2_size)]
        self.v_W3 = [[0.0] * output_size for _ in range(hidden2_size)]
        self.m_b3 = [0.0] * output_size
        self.v_b3 = [0.0] * output_size

        self.t = 0  # contador de Adam

    def _initialize_weights(self, fan_in, fan_out):
        """Inicialización Xavier/Glorot para pesos."""
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return [[random.uniform(-limit, limit) for _ in range(fan_out)] for _ in range(fan_in)]

    def _dropout(self, layer, rate):
        """Aplica dropout durante entrenamiento."""
        if not self.training:
            return layer
        mask = [1.0 if random.random() > rate else 0.0 for _ in layer]
        return [x * m / (1.0 - rate) for x, m in zip(layer, mask)]

    def forward(self, X, training=True):
        """Forward pass con dropout. X debe ser una lista de floats (una muestra)."""
        self.training = training

        # Capa 1: Input -> Hidden1
        self.z1 = []
        for i in range(self.hidden1_size):
            z1_i = self.b1[i]
            for j in range(self.input_size):
                z1_i += X[j] * self.W1[j][i]
            self.z1.append(z1_i)

        self.a1 = [relu(z) for z in self.z1]
        if training:
            self.a1 = self._dropout(self.a1, self.dropout_rate)

        # Capa 2: Hidden1 -> Hidden2
        self.z2 = []
        for i in range(self.hidden2_size):
            z2_i = self.b2[i]
            for j in range(self.hidden1_size):
                z2_i += self.a1[j] * self.W2[j][i]
            self.z2.append(z2_i)

        self.a2 = [relu(z) for z in self.z2]
        if training:
            self.a2 = self._dropout(self.a2, self.dropout_rate)

        # Capa 3: Hidden2 -> Output
        self.z3 = []
        for i in range(self.output_size):
            z3_i = self.b3[i]
            for j in range(self.hidden2_size):
                z3_i += self.a2[j] * self.W3[j][i]
            self.z3.append(z3_i)

        self.a3 = [sigmoid(z) for z in self.z3]
        return self.a3

    def backward(self, X, y):
        """Backward pass con regularización L2."""
        # Calcular errores
        delta3 = [(a - target) for a, target in zip(self.a3, y)]

        # Gradientes capa 3
        dW3 = [[0.0] * self.output_size for _ in range(self.hidden2_size)]
        db3 = [0.0] * self.output_size

        for i in range(self.hidden2_size):
            for j in range(self.output_size):
                dW3[i][j] = delta3[j] * self.a2[i] + self.l2_lambda * self.W3[i][j]
                if i == 0:
                    db3[j] = delta3[j]

        # Gradientes capa 2
        delta2 = [0.0] * self.hidden2_size
        for i in range(self.hidden2_size):
            for j in range(self.output_size):
                delta2[i] += delta3[j] * self.W3[i][j]
            delta2[i] *= relu_derivative(self.z2[i])

        dW2 = [[0.0] * self.hidden2_size for _ in range(self.hidden1_size)]
        db2 = [0.0] * self.hidden2_size

        for i in range(self.hidden1_size):
            for j in range(self.hidden2_size):
                dW2[i][j] = delta2[j] * self.a1[i] + self.l2_lambda * self.W2[i][j]
                if i == 0:
                    db2[j] = delta2[j]

        # Gradientes capa 1
        delta1 = [0.0] * self.hidden1_size
        for i in range(self.hidden1_size):
            for j in range(self.hidden2_size):
                delta1[i] += delta2[j] * self.W2[i][j]
            delta1[i] *= relu_derivative(self.z1[i])

        dW1 = [[0.0] * self.hidden1_size for _ in range(self.input_size)]
        db1 = [0.0] * self.hidden1_size

        for i in range(self.input_size):
            for j in range(self.hidden1_size):
                dW1[i][j] = delta1[j] * X[i] + self.l2_lambda * self.W1[i][j]
                if i == 0:
                    db1[j] = delta1[j]

        return dW1, db1, dW2, db2, dW3, db3

    def _adam_update(self, param, grad, m, v, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Actualización Adam para un parámetro."""
        self.t += 1
        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m_new / (1 - beta1 ** self.t)
        v_hat = v_new / (1 - beta2 ** self.t)

        param_new = param - self.learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)
        return param_new, m_new, v_new

    def update_parameters(self, dW1, db1, dW2, db2, dW3, db3):
        """Actualiza parámetros usando Adam."""
        # Actualizar W1 y b1
        for i in range(self.input_size):
            for j in range(self.hidden1_size):
                self.W1[i][j], self.m_W1[i][j], self.v_W1[i][j] = self._adam_update(
                    self.W1[i][j], dW1[i][j], self.m_W1[i][j], self.v_W1[i][j])

        for j in range(self.hidden1_size):
            self.b1[j], self.m_b1[j], self.v_b1[j] = self._adam_update(
                self.b1[j], db1[j], self.m_b1[j], self.v_b1[j])

        # Actualizar W2 y b2
        for i in range(self.hidden1_size):
            for j in range(self.hidden2_size):
                self.W2[i][j], self.m_W2[i][j], self.v_W2[i][j] = self._adam_update(
                    self.W2[i][j], dW2[i][j], self.m_W2[i][j], self.v_W2[i][j])

        for j in range(self.hidden2_size):
            self.b2[j], self.m_b2[j], self.v_b2[j] = self._adam_update(
                self.b2[j], db2[j], self.m_b2[j], self.v_b2[j])

        # Actualizar W3 y b3
        for i in range(self.hidden2_size):
            for j in range(self.output_size):
                self.W3[i][j], self.m_W3[i][j], self.v_W3[i][j] = self._adam_update(
                    self.W3[i][j], dW3[i][j], self.m_W3[i][j], self.v_W3[i][j])

        for j in range(self.output_size):
            self.b3[j], self.m_b3[j], self.v_b3[j] = self._adam_update(
                self.b3[j], db3[j], self.m_b3[j], self.v_b3[j])

    def fit(self, X, y, epochs=10000, batch_size=32, validation_split=0.2, patience=100):
        """Entrena la red neuronal con early stopping y mini-batches."""
        n_samples = len(X)

        # Dividir en train/validation
        val_size = int(n_samples * validation_split)
        train_size = n_samples - val_size

        # Shuffle data
        indices = list(range(n_samples))
        random.shuffle(indices)
        X_train = [X[i] for i in indices[:train_size]]
        y_train = [y[i] for i in indices[:train_size]]
        X_val = [X[i] for i in indices[train_size:]]
        y_val = [y[i] for i in indices[train_size:]]

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        print(f"Entrenando red neuronal avanzada: {epochs} epochs, {len(X_train)} muestras training, {len(X_val)} validation")

        for epoch in range(epochs):
            # Mini-batch training
            epoch_loss = 0.0
            n_batches = len(X_train) // batch_size

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))

                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]

                # Forward pass
                predictions = []
                for x in batch_X:
                    pred = self.forward(x, training=True)
                    predictions.extend(pred)

                # Backward pass
                total_dW1 = [[0.0] * self.hidden1_size for _ in range(self.input_size)]
                total_db1 = [0.0] * self.hidden1_size
                total_dW2 = [[0.0] * self.hidden2_size for _ in range(self.hidden1_size)]
                total_db2 = [0.0] * self.hidden2_size
                total_dW3 = [[0.0] * self.output_size for _ in range(self.hidden2_size)]
                total_db3 = [0.0] * self.output_size

                for x, target in zip(batch_X, batch_y):
                    dW1, db1, dW2, db2, dW3, db3 = self.backward(x, [target])

                    for i in range(len(total_dW1)):
                        for j in range(len(total_dW1[i])):
                            total_dW1[i][j] += dW1[i][j]
                    for i in range(len(total_db1)):
                        total_db1[i] += db1[i]

                    for i in range(len(total_dW2)):
                        for j in range(len(total_dW2[i])):
                            total_dW2[i][j] += dW2[i][j]
                    for i in range(len(total_db2)):
                        total_db2[i] += db2[i]

                    for i in range(len(total_dW3)):
                        for j in range(len(total_dW3[i])):
                            total_dW3[i][j] += dW3[i][j]
                    for i in range(len(total_db3)):
                        total_db3[i] += db3[i]

                # Average gradients
                batch_size_actual = len(batch_X)
                for i in range(len(total_dW1)):
                    for j in range(len(total_dW1[i])):
                        total_dW1[i][j] /= batch_size_actual
                for i in range(len(total_db1)):
                    total_db1[i] /= batch_size_actual

                for i in range(len(total_dW2)):
                    for j in range(len(total_dW2[i])):
                        total_dW2[i][j] /= batch_size_actual
                for i in range(len(total_db2)):
                    total_db2[i] /= batch_size_actual

                for i in range(len(total_dW3)):
                    for j in range(len(total_dW3[i])):
                        total_dW3[i][j] /= batch_size_actual
                for i in range(len(total_db3)):
                    total_db3[i] /= batch_size_actual

                # Update parameters
                self.update_parameters(total_dW1, total_db1, total_dW2, total_db2, total_dW3, total_db3)

                # Calculate batch loss
                for pred, target in zip(predictions, batch_y):
                    loss = -target * math.log(pred + 1e-15) - (1 - target) * math.log(1 - pred + 1e-15)
                    epoch_loss += loss

            epoch_loss /= len(X_train)

            # Validation
            if epoch % 10 == 0:
                val_predictions = []
                for x in X_val:
                    pred = self.forward(x, training=False)
                    val_predictions.extend(pred)

                val_loss = 0.0
                for pred, target in zip(val_predictions, y_val):
                    loss = -target * math.log(pred + 1e-15) - (1 - target) * math.log(1 - pred + 1e-15)
                    val_loss += loss
                val_loss /= len(X_val)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Guardar mejores pesos
                    best_weights = {
                        'W1': [row[:] for row in self.W1],
                        'b1': self.b1[:],
                        'W2': [row[:] for row in self.W2],
                        'b2': self.b2[:],
                        'W3': [row[:] for row in self.W3],
                        'b3': self.b3[:]
                    }
                else:
                    patience_counter += 1

                if epoch % 100 == 0:
                    print(".4f")

                if patience_counter >= patience:
                    print(f"Early stopping en epoch {epoch}, mejor val_loss: {best_val_loss:.4f}")
                    # Restaurar mejores pesos
                    if best_weights:
                        self.W1 = best_weights['W1']
                        self.b1 = best_weights['b1']
                        self.W2 = best_weights['W2']
                        self.b2 = best_weights['b2']
                        self.W3 = best_weights['W3']
                        self.b3 = best_weights['b3']
                    break

    def predict_proba(self, X):
        """Predice probabilidades."""
        predictions = []
        for x in X:
            pred = self.forward(x, training=False)
            predictions.append(pred[0])
        return predictions

    def predict(self, X, threshold=0.5):
        """Predice clases binarias."""
        probs = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probs]

    def predict_diabetes_stage(self, features):
        """
        Predice etapa de diabetes: 0=Normal, 1=Prediabetes, 2=Diabetes
        """
        prob = self.predict_proba([features])[0]
        glucose_fast = features[1]
        glucose_post = features[2]
        a1c = features[3]

        # Lógica de clasificación por etapas con umbrales más precisos
        if prob >= 0.75 or glucose_fast >= 126 or glucose_post >= 200 or a1c >= 6.5:
            return 2  # Diabetes
        elif prob >= 0.4 or (glucose_fast >= 100 and glucose_fast < 126) or (glucose_post >= 140 and glucose_post < 200) or (a1c >= 5.7 and a1c < 6.5):
            return 1  # Prediabetes
        else:
            return 0  # Normal


class AdvancedDiabetesModel(AdvancedNeuralNetwork):
    """
    Modelo avanzado de diabetes basado en red neuronal profunda.
    Hereda de AdvancedNeuralNetwork con configuración específica para diabetes.
    """

    def __init__(self, n_features):
        # Configuración optimizada para diabetes: 15 features -> 64 -> 32 -> 1
        super().__init__(
            input_size=n_features,
            hidden1_size=64,
            hidden2_size=32,
            output_size=1,
            learning_rate=0.001,
            l2_lambda=0.001,
            dropout_rate=0.2
        )

        self.n = n_features
        self.feature_names = [
            "Edad", "Glucosa ayuno", "Glucosa 2h", "A1C", "IMC",
            "LDL", "HDL", "Triglicéridos", "Colesterol Total",
            "Presión Sistólica", "Presión Diastólica", "Cuidados",
            "Antecedentes Familiares", "Ejercicio Semanal", "Dieta Calidad"
        ]


class AdvancedDiabetesModel(AdvancedNeuralNetwork):
    """
    Modelo avanzado de diabetes basado en red neuronal profunda.
    Hereda de AdvancedNeuralNetwork con configuración específica para diabetes.
    """

    def __init__(self, n_features):
        # Configuración optimizada para diabetes: 15 features -> 64 -> 32 -> 1
        super().__init__(
            input_size=n_features,
            hidden1_size=64,
            hidden2_size=32,
            output_size=1,
            learning_rate=0.001,
            l2_lambda=0.001,
            dropout_rate=0.2
        )

        self.n = n_features
        self.feature_names = [
            "Edad", "Glucosa ayuno", "Glucosa 2h", "A1C", "IMC",
            "LDL", "HDL", "Triglicéridos", "Colesterol Total",
            "Presión Sistólica", "Presión Diastólica", "Cuidados",
            "Antecedentes Familiares", "Ejercicio Semanal", "Dieta Calidad"
        ]


def get_age_prevalence_factor(age):
    """
    Retorna factor de prevalencia basado en grupo de edad y tipo de diabetes.
    Mejorado con datos epidemiológicos actuales que consideran:
    - Diabetes Tipo 1: picos entre 4-7 años y 10-14 años
    - Diabetes Tipo 2: >45 años (tradicional) pero creciente en 20-30 años y adolescentes
    - Factores: genética, sobrepeso, sedentarismo
    """
    if age < 4:
        return 0.001  # Muy baja en preescolares
    elif 4 <= age <= 7:
        return 0.03   # PICO 1: Diabetes Tipo 1, sospecha si hay malnutrición
    elif 8 <= age <= 10:
        return 0.015  # Baja entre picos
    elif 10 <= age <= 14:
        return 0.035  # PICO 2: Diabetes Tipo 1 y emergencia de Tipo 2 adolescente
    elif 15 <= age <= 19:
        return 0.025  # Disminuye después de picos pediátricos
    elif 20 <= age <= 30:
        return 0.08   # AUMENTA: Emergencia de Tipo 2 en jóvenes adultos
    elif 31 <= age <= 44:
        return 0.12   # Sigue aumentando pre-45
    elif 45 <= age <= 54:
        return 0.18   # Riesgo alto (rango tradicional Tipo 2)
    elif 55 <= age <= 64:
        return 0.22   # Muy alto
    else:  # >= 65
        return 0.30   # Máxima prevalencia (1 de cada 3)


def is_childhood_prediabetes(age, glucose_fast, glucose_post, a1c):
    """
    Evalúa prediabetes infantil y diabetes Tipo 1 temprana.
    Considerando picos epidemiológicos en 4-7 y 10-14 años.
    """
    # Rango pico 1: Diabetes Tipo 1 (4-7 años)
    if 4 <= age <= 7:
        # Criterios más sensibles para detección Tipo 1 temprana
        if glucose_fast >= 100 or glucose_post >= 140 or a1c >= 5.7:
            return True
    
    # Rango pico 2: Diabetes Tipo 1 y Tipo 2 emergente (10-14 años)
    elif 10 <= age <= 14:
        # Criterios para ambos tipos
        if glucose_fast >= 100 or glucose_post >= 140 or a1c >= 5.7:
            return True
    
    # Rango intermedio (8-9 años): criterios más estrictos
    elif 8 <= age <= 9:
        if glucose_fast >= 110 or glucose_post >= 150 or a1c >= 5.9:
            return True
    
    # Adolescentes tempranos (15-19 años): vigilancia para Tipo 2
    elif 15 <= age <= 19:
        # Criterios para Tipo 2 emergente en jóvenes
        if glucose_fast >= 100 or glucose_post >= 140 or a1c >= 5.7:
            return True
    
    return False


def calculate_bmi_risk(bmi):
    """
    Calcula factor de riesgo por IMC."""
    if bmi < 18.5:
        return 0.8  # Bajo peso, menor riesgo
    elif bmi < 25:
        return 1.0  # Normal
    elif bmi < 30:
        return 1.5  # Sobrepeso
    elif bmi < 35:
        return 2.0  # Obesidad I
    else:
        return 2.5  # Obesidad II+


def is_type2_early_onset(age, glucose_fast, glucose_post, a1c, bmi, family_history, exercise_weekly):
    """
    Detecta Diabetes Tipo 2 de inicio temprano en jóvenes (20-44 años).
    Basado en la emergencia actual de Tipo 2 en adultos jóvenes.
    
    Factores considerados:
    - Adultos jóvenes (20-44) con factores de riesgo combinados
    - Sobrepeso/Obesidad (BMI >= 25)
    - Bajo nivel de actividad física
    - Antecedentes familiares
    - Criterios de prediabetes/diabetes
    """
    if not (20 <= age <= 44):
        return False
    
    # Contar factores de riesgo presentes
    risk_factors = 0
    
    # Factor 1: Sobrepeso/Obesidad
    if bmi >= 25:
        risk_factors += 1
    
    # Factor 2: Sedentarismo
    if exercise_weekly < 150:  # Menos actividad recomendada (150 min/semana)
        risk_factors += 1
    
    # Factor 3: Antecedentes familiares
    if family_history:
        risk_factors += 1
    
    # Factor 4: Valores de glucosa elevados
    if glucose_fast >= 100 or glucose_post >= 140 or a1c >= 5.7:
        risk_factors += 1
    
    # Diagnóstico: Si hay múltiples factores + valores de glucosa elevados
    # Se considera posible Tipo 2 temprana
    if risk_factors >= 3 and (glucose_fast >= 100 or glucose_post >= 140 or a1c >= 5.7):
        return True
    
    # O si hay diagnóstico claro de prediabetes/diabetes con cualquier otro factor
    if (glucose_fast >= 125 or glucose_post >= 140 or a1c >= 5.7) and risk_factors >= 2:
        return True
    
    return False


def generate_advanced_dataset(n_samples=5000, seed=42):
    """
    Genera dataset sintético avanzado con 15 características.
    Incluye factores de riesgo más precisos y prevalencia por edad.
    Versión mejorada con distribución realista considerando:
    - Picos de Tipo 1: 4-7 y 10-14 años
    - Emergencia de Tipo 2: 20-30 años
    - Riesgo máximo: 45+ años
    """
    random.seed(seed)
    X = []
    y = []

    print(f"Generando {n_samples} muestras de datos sintéticos...")

    for i in range(n_samples):
        if i % 1000 == 0:
            print(f"Procesando muestra {i+1}/{n_samples}")

        # Características demográficas con distribución mejorada
        # Distribución mixta para cubrir picos epidemiológicos
        age_distribution = random.random()
        if age_distribution < 0.10:  # 10% niños con picos Tipo 1
            age = random.choice([random.gauss(5.5, 1.5), random.gauss(12, 2)])  # Picos 4-7 y 10-14
            age = max(4, min(14, age))
        elif age_distribution < 0.20:  # 10% jóvenes adultos (20-30)
            age = random.gauss(25, 5)
            age = max(20, min(30, age))
        else:  # 80% distribución general
            age = random.gauss(47, 22)
            age = max(8, min(90, age))

        # Distribución de IMC más realista según edad
        if age < 20:
            bmi_base = random.gauss(22, 4)  # Jóvenes más delgados
        elif age < 30:
            bmi_base = random.gauss(24, 4.5)  # Adultos jóvenes
        elif age < 50:
            bmi_base = random.gauss(26, 5)  # Adultos medios
        else:
            bmi_base = random.gauss(28, 6)  # Mayores con tendencia a sobrepeso

        bmi = max(15, min(50, bmi_base))

        # Características glucémicas con mayor realismo
        # Base de glucosa que aumenta con edad y BMI
        base_glucose = 85 + (age * 0.2) + (bmi - 25) * 0.8

        # Añadir variabilidad individual
        glucose_variability = random.gauss(0, 15)
        avg_fast = max(60, min(300, base_glucose + glucose_variability))

        # Glucosa post-prandial correlacionada con ayuno
        post_variability = random.gauss(0, 25)
        avg_post = max(80, min(400, avg_fast + 50 + post_variability))

        # A1C correlacionada con glucosa promedio
        a1c_base = 4.5 + ((avg_fast + avg_post)/2 - 100) * 0.02
        a1c = max(4.0, min(12.0, a1c_base + random.gauss(0, 0.5)))

        # Características lipídicas mejoradas con correlaciones realistas
        # LDL aumenta con edad y BMI
        ldl_base = 90 + (age * 0.3) + (bmi - 25) * 1.2
        ldl = max(50, min(250, ldl_base + random.gauss(0, 25)))

        # HDL generalmente mejor en jóvenes y mujeres (simplificado)
        hdl_base = 55 - (age * 0.1) + random.gauss(0, 10)
        hdl = max(25, min(100, hdl_base))

        # Triglicéridos correlacionados con BMI y glucosa
        trig_base = 120 + (bmi - 25) * 3 + (avg_fast - 100) * 0.5
        triglycerides = max(50, min(400, trig_base + random.gauss(0, 30)))

        total_chol = ldl + hdl + (triglycerides / 5.0) + random.gauss(0, 10)

        # Presión arterial con mayor realismo
        # Hipertensión más común en mayores y obesos
        bp_base = 110 + (age * 0.4) + (bmi - 25) * 0.8
        systolic = max(90, min(200, bp_base + random.gauss(0, 12)))
        diastolic = max(60, min(120, systolic * 0.6 + random.gauss(0, 8)))

        # Factores de estilo de vida con distribuciones más realistas
        exercise_weekly = random.gauss(150, 120)  # Más gente sedentaria
        exercise_weekly = max(0, min(600, exercise_weekly))

        # Calidad de dieta: distribución sesgada hacia calidades medias-bajas
        diet_quality = max(1, min(5, int(random.gauss(2.5, 1.2))))

        # Antecedentes familiares con mayor realismo epidemiológico
        family_risk = 0
        if random.random() < 0.25:  # 25% tienen antecedentes familiares
            family_risk = 1

        # Calcular care_score mejorado con más factores
        care_score = 3.0  # Base neutral

        # Ajustes por ejercicio
        if exercise_weekly >= 300:
            care_score -= 1.0  # Muy activo
        elif exercise_weekly >= 150:
            care_score -= 0.5  # Moderadamente activo
        elif exercise_weekly < 30:
            care_score += 0.5  # Sedentario

        # Ajustes por dieta
        care_score -= (diet_quality - 3) * 0.3  # Mejor dieta = mejor score

        # Ajustes por otros factores
        if bmi >= 35:
            care_score += 1.0  # Obesidad severa
        elif bmi >= 30:
            care_score += 0.7
        elif bmi >= 25:
            care_score += 0.3

        if systolic >= 160 or diastolic >= 100:
            care_score += 0.8  # Hipertensión severa
        elif systolic >= 140 or diastolic >= 90:
            care_score += 0.4

        care_score += family_risk * 0.6  # Antecedentes familiares
        care_score = max(0.0, min(5.0, care_score))

        # Score de riesgo avanzado con mayor complejidad
        age_factor = get_age_prevalence_factor(age)
        bmi_factor = calculate_bmi_risk(bmi)

        # Score principal con múltiples interacciones
        score = (
            # Factores glucémicos (peso principal)
            0.08 * (avg_fast - 90) +
            0.06 * (avg_post - 120) +
            1.5 * (a1c - 5.0) +

            # Factores antropométricos
            0.03 * (age - 40) +
            0.2 * (bmi_factor - 1.0) +

            # Factores lipídicos
            0.01 * (ldl - 100) +
            (-0.02) * (hdl - 50) +
            0.012 * (triglycerides - 150) +

            # Factores cardiovasculares
            0.007 * (systolic - 120) +
            0.01 * (diastolic - 80) +

            # Factores de estilo de vida
            0.3 * care_score +
            0.8 * family_risk +

            # Factor epidemiológico
            3.0 * (age_factor - 0.045)
        )

        # Interacciones no lineales
        # Mayor riesgo cuando múltiples factores se combinan
        interaction_bonus = 0.0
        high_risk_factors = sum([
            avg_fast >= 110, avg_post >= 160, a1c >= 6.0,
            bmi >= 30, ldl >= 140, triglycerides >= 200,
            systolic >= 140, family_risk == 1
        ])

        if high_risk_factors >= 3:
            interaction_bonus = high_risk_factors * 0.2

        score += interaction_bonus

        # Ajuste especial para prediabetes infantil
        if is_childhood_prediabetes(age, avg_fast, avg_post, a1c):
            score += 1.5  # Mayor peso para casos pediátricos

        # Probabilidad con sigmoid y ajuste final
        prob = sigmoid(score)

        # Etiqueta con algo de ruido para realismo
        noise = random.gauss(0, 0.1)  # 10% de ruido
        final_prob = max(0, min(1, prob + noise))
        label = 1 if random.random() < final_prob else 0

        features = [
            age, avg_fast, avg_post, a1c, bmi,
            ldl, hdl, triglycerides, total_chol,
            systolic, diastolic, care_score,
            family_risk, exercise_weekly, diet_quality
        ]

        X.append(features)
        y.append(label)

    print(f"Dataset completado: {len(X)} muestras generadas")
    return X, y


def train_advanced_model():
    """Entrena modelo avanzado con red neuronal profunda y 15 características."""
    print("Generando dataset avanzado de entrenamiento...")
    X, y = generate_advanced_dataset(5000)  # Aumentado a 5000 muestras
    print(f"Dataset generado: {len(X)} muestras con {len(X[0])} características")

    model = AdvancedDiabetesModel(n_features=15)
    print("Entrenando red neuronal profunda...")
    model.fit(X, y, epochs=2000, batch_size=64, validation_split=0.2, patience=50)  # Parámetros optimizados
    print("Entrenamiento completado.")
    return model


def assess_diabetes_risk_comprehensive(features, model):
    """
    Evaluación integral del riesgo de diabetes.
    Retorna: (probabilidad, etapa, factores_riesgo, recomendaciones)
    """
    prob = model.predict_proba([features])[0]
    stage = model.predict_diabetes_stage(features)

    age, glucose_fast, glucose_post, a1c, bmi = features[:5]
    ldl, hdl, triglycerides, total_chol = features[5:9]
    systolic, diastolic = features[9:11]
    care_score, family_history, exercise_weekly, diet_quality = features[11:15]

    # Evaluar factores de riesgo
    risk_factors = []

    # Glucosa
    if glucose_fast >= 126 or glucose_post >= 200 or a1c >= 6.5:
        risk_factors.append("Glucosa elevada (Diabetes)")
    elif glucose_fast >= 100 or glucose_post >= 140 or a1c >= 5.7:
        risk_factors.append("Glucosa borderline (Prediabetes)")

    # Edad y prediabetes infantil
    if is_childhood_prediabetes(age, glucose_fast, glucose_post, a1c):
        risk_factors.append("Prediabetes infantil (4-7 y 10-14 años)")
    
    # Detección de Diabetes Tipo 2 temprana en jóvenes adultos
    if is_type2_early_onset(age, glucose_fast, glucose_post, a1c, bmi, family_history, exercise_weekly):
        risk_factors.append("Diabetes Tipo 2 temprana (jóvenes adultos 20-44 años)")

    # IMC
    if bmi >= 30:
        risk_factors.append("Obesidad")
    elif bmi >= 25:
        risk_factors.append("Sobrepeso")

    # Colesterol
    if ldl > 160:
        risk_factors.append("LDL muy elevado")
    if hdl < 40:
        risk_factors.append("HDL bajo")
    if triglycerides > 200:
        risk_factors.append("Triglicéridos elevados")

    # Presión arterial
    if systolic >= 140 or diastolic >= 90:
        risk_factors.append("Hipertensión")

    # Antecedentes
    if family_history:
        risk_factors.append("Antecedentes familiares")
    
    # Estilo de vida
    if exercise_weekly < 150:
        risk_factors.append("Sedentarismo (< 150 min/semana)")

    # Estilo de vida
    if exercise_weekly < 150:
        risk_factors.append("Sedentarismo")
    if diet_quality <= 2:
        risk_factors.append("Dieta pobre")

    # Generar recomendaciones
    recommendations = generate_comprehensive_recommendations(
        stage, age, bmi, ldl, hdl, triglycerides, exercise_weekly, diet_quality
    )

    return prob, stage, risk_factors, recommendations


def generate_comprehensive_recommendations(stage, age, bmi, ldl, hdl, triglycerides, exercise, diet_quality):
    """Genera recomendaciones integrales basadas en todos los factores."""
    recommendations = []

    # Recomendaciones por etapa
    if stage == 2:  # Diabetes
        recommendations.append("• Consulta inmediata con endocrinólogo")
        recommendations.append("• Iniciar tratamiento médico si no se ha hecho")
        recommendations.append("• Monitoreo continuo de glucosa")
    elif stage == 1:  # Prediabetes
        recommendations.append("• Cambios en estilo de vida para prevenir diabetes")
        recommendations.append("• Seguimiento médico cada 3-6 meses")
        if age <= 12:
            recommendations.append("• Atención especial pediátrica para prediabetes infantil")
        elif 20 <= age <= 44:
            recommendations.append("• Atención para Diabetes Tipo 2 temprana en jóvenes adultos")
            recommendations.append("• Intensificar cambios de estilo de vida: dieta y ejercicio")

    # Recomendaciones específicas para jóvenes adultos (20-44) con riesgo
    if 20 <= age <= 44 and stage >= 1:
        recommendations.append("• Intensificar combate contra sedentarismo: 150+ min ejercicio/semana")
        recommendations.append("• Evaluación nutricional especializada")
        recommendations.append("• Control de antecedentes familiares")

    # Recomendaciones por IMC
    if bmi >= 35:
        recommendations.append("• Evaluación bariátrica")
        recommendations.append("• Programa de pérdida de peso supervisado")
    elif bmi >= 30:
        recommendations.append("• Pérdida de peso gradual (0.5-1 kg/mes)")
    elif bmi >= 25:
        recommendations.append("• Mantener peso estable")

    # Recomendaciones cardiovasculares
    if ldl > 160 or hdl < 40 or triglycerides > 200:
        recommendations.append("• Control lipídico con cardiólogo")
        recommendations.append("• Dieta cardioprotectora")

    # Recomendaciones de ejercicio
    if exercise < 150:
        recommendations.append("• Aumentar actividad física a 150 min/semana")
        recommendations.append("• Caminar 30 min diarios como mínimo")

    # Recomendaciones dietéticas
    if diet_quality <= 2:
        recommendations.append("• Consulta con nutricionista")
        recommendations.append("• Dieta mediterránea o DASH")

    # Recomendaciones por edad
    if age <= 12 and stage >= 1:
        recommendations.append("• Seguimiento pediátrico especializado")
        recommendations.append("• Educación familiar sobre diabetes")
    elif 15 <= age <= 19 and stage >= 1:
        recommendations.append("• Monitoreo especial en adolescentes")
        recommendations.append("• Educación sobre hábitos de vida saludables")
    elif 20 <= age <= 44 and stage >= 1:
        recommendations.append("• Atención especial: Diabetes Tipo 2 emergente en adultos jóvenes")
        recommendations.append("• Evaluación genética si hay antecedentes familiares")
    elif age >= 65:
        recommendations.append("• Adaptar recomendaciones a condiciones geriátricas")
        recommendations.append("• Evaluación de hipoglucemia y medicamentos")

    if not recommendations:
        recommendations.append("• Mantener hábitos saludables")
        recommendations.append("• Chequeos anuales de rutina")

    return recommendations


def get_stage_description(stage):
    """Retorna descripción de la etapa de diabetes."""
    descriptions = {
        0: "Normal - Sin evidencia de diabetes o prediabetes",
        1: "Prediabetes - Riesgo elevado de desarrollar diabetes tipo 2",
        2: "Diabetes - Requiere manejo médico activo"
    }
    return descriptions.get(stage, "Indeterminado")


def calculate_precision_score(features, model):
    """
    Calcula score de precisión basado en consistencia de factores de riesgo.
    Retorna valor entre 0-1 donde 1 es máxima consistencia.
    """
    age, glucose_fast, glucose_post, a1c, bmi = features[:5]
    ldl, hdl, triglycerides = features[5:8]
    systolic, diastolic = features[9:11]
    care_score, family_history = features[11:13]

    consistency_factors = 0
    total_factors = 0

    # Consistencia glucosa-A1C
    total_factors += 1
    if (glucose_fast >= 126 and a1c >= 6.5) or (glucose_fast < 100 and a1c < 5.7):
        consistency_factors += 1

    # Consistencia IMC-presión
    total_factors += 1
    if (bmi >= 30 and (systolic >= 130 or diastolic >= 85)) or (bmi < 25 and systolic < 120 and diastolic < 80):
        consistency_factors += 1

    # Consistencia colesterol
    total_factors += 1
    lipid_risk = (ldl > 130) + (hdl < 50) + (triglycerides > 150)
    if lipid_risk >= 2 or lipid_risk == 0:
        consistency_factors += 1

    return consistency_factors / total_factors if total_factors > 0 else 0.5




def calculate_family_risk_factor():
    """
    Calcula factor de riesgo familiar basado en preguntas detalladas.
    Retorna factor multiplicador (1.0-3.0) donde mayor valor = mayor riesgo.
    """
    # Esta función será llamada desde Proyecto.py
    # Retorna 1.0 por defecto, será sobrescrita por las preguntas
    return 1.0


def get_family_history_details():
    """
    Función auxiliar para obtener detalles de antecedentes familiares.
    Retorna diccionario con información detallada.
    """
    return {
        'has_family_history': False,
        'family_count': 0,
        'relationship_types': [],
        'average_onset_age': None,
        'risk_factor': 1.0
    }


def detect_gestational_diabetes(age_female, weeks_pregnant, glucose_fast, glucose_post, bmi, family_history):
    """
    Detecta riesgo de Diabetes Gestacional durante embarazo (semanas 24-28).
    
    Criterios:
    - Edad: mujeres en edad reproductiva (15-50 años aproximadamente)
    - Embarazo: detectado cuando se especifica semanas_pregnant entre 8-40 semanas
    - Valores de glucosa elevados para embarazo (criterios más estrictos que normales)
    - Factores de riesgo: sobrepeso, antecedentes familiares
    
    Retorna: (is_gestational_diabetes, risk_level, semana_critica)
    """
    # Validación de rango de edad reproductiva
    if age_female < 15 or age_female > 50:
        return False, "No aplicable", None
    
    # Detector de embarazo: si weeks_pregnant está entre 8 y 40 semanas
    if not (8 <= weeks_pregnant <= 40):
        return False, "No embarazada", None
    
    # Semana crítica para screening (entre semanas 24-28)
    is_critical_week = (24 <= weeks_pregnant <= 28)
    
    # Criterios de glucosa para diabetes gestacional (más estrictos que normales)
    # Valores en ayunas: >= 92 mg/dL
    # Valores a las 2h: >= 153 mg/dL (algunos usan 155)
    has_elevated_glucose = (glucose_fast >= 92) or (glucose_post >= 153)
    
    # Factores de riesgo adicionales
    has_risk_factors = (bmi >= 25) or family_history
    
    # Diagnóstico de diabetes gestacional
    if has_elevated_glucose:
        if is_critical_week:
            # En semana crítica con glucosa elevada = diagnóstico probable
            risk_level = "Alto (en semana crítica de screening)"
            return True, risk_level, weeks_pregnant
        elif weeks_pregnant > 28:
            # Después del screening, si hay glucosa elevada = diagnóstico
            risk_level = "Diagnóstico (diabetes gestacional confirmada)"
            return True, risk_level, weeks_pregnant
        else:
            # Antes del screening pero con glucosa elevada
            if has_risk_factors:
                risk_level = "Moderado-Alto (pre-screening, con factores de riesgo)"
            else:
                risk_level = "Moderado (pre-screening)"
            return True, risk_level, weeks_pregnant
    
    # Sin glucosa elevada, pero en semana crítica con factores de riesgo
    if is_critical_week and has_risk_factors:
        return False, "Riesgo moderado (screening recomendado)", weeks_pregnant
    
    return False, "Bajo riesgo", None


def possible_symptoms_by_probability(prob, stage=None, age=None):
    """Retorna síntomas posibles según nivel de riesgo, etapa y edad."""

    # Síntomas específicos para prediabetes infantil
    if stage == 1 and age is not None and age <= 12:
        return [
            "(*) Aumento de sed y orina frecuente",
            "(*) Fatiga después de comidas",
            "(*) Cambios en el apetito",
            "(*) Dificultad de concentración",
            "(*) Irritabilidad o cambios de humor",
            "(*) Infecciones recurrentes",
            "(*) Crecimiento acelerado (obesidad)"
        ]

    # Síntomas por etapa y probabilidad
    if stage == 2 or prob >= 0.75:  # Diabetes establecida o alto riesgo
        return [
            "(*) Sed excesiva y constante",
            "(*) Orinar frecuentemente, incluso por la noche",
            "(*) Fatiga severa e inexplicable",
            "(*) Visión borrosa o cambios en la vista",
            "(*) Cicatrización lenta de heridas",
            "(*) Hormigueo o entumecimiento en extremidades",
            "(*) Infecciones recurrentes",
            "(*) Pérdida de peso inexplicada"
        ]
    elif stage == 1 or prob >= 0.6:  # Prediabetes o riesgo alto
        return [
            "(*) Mayor sed de lo normal",
            "(*) Necesidad más frecuente de orinar",
            "(*) Fatiga ocasional después de comer",
            "(*) Náusea o cambios en el apetito",
            "(*) Dolores de cabeza frecuentes",
            "(*) Dificultad para concentrarse",
            "(*) Aumento leve de peso"
        ]
    elif prob >= 0.4:  # Riesgo moderado
        return [
            "(*) Pequeñas variaciones de energía durante el día",
            "(*) Hambre intermitente o antojo de dulces",
            "(*) Leve aumento de sed",
            "(*) Pequeños cambios en el peso",
            "(*) Irritabilidad ocasional",
            "(*) Sueño alterado"
        ]
    else:  # Riesgo bajo
        return [
            "(*) Bajo riesgo: pocos síntomas esperables",
            "(*) Mantener hábitos saludables (ejercicio, dieta equilibrada)",
            "(*) Revisar glucosa y colesterol regularmente",
            "(*) Chequeos preventivos anuales"
        ]


def map_probability_to_timeframe(prob, age=None, avg_glucose=None, ldl=None, hdl=None, stage=None, family_risk_factor=1.0):
    """
    Calcula timeframe coherente basado en probabilidad, factores clínicos y etapa.
    Basado en datos epidemiológicos reales de progresión diabetes.

    Args:
        prob: Probabilidad de desarrollar diabetes (0-1)
        age: Edad del paciente
        avg_glucose: Glucosa promedio
        ldl: Colesterol LDL
        hdl: Colesterol HDL
        stage: Etapa (0=Normal, 1=Prediabetes, 2=Diabetes)
        family_risk_factor: Factor de ajuste por antecedentes familiares (1.0-3.0)

    Returns:
        String con timeframe y clasificación de riesgo
    """

    if age is None or avg_glucose is None:
        # Fallback a timeframe simple si no hay datos clínicos
        if prob >= 0.75:
            return "< 1 año (Riesgo Crítico - Control urgente)"
        if prob >= 0.6:
            return "1-2 años (Riesgo Alto)"
        if prob >= 0.4:
            return "3-5 años (Riesgo Moderado)"
        if prob >= 0.25:
            return "5-8 años (Riesgo Bajo)"
        return "> 10 años (Riesgo Mínimo)"

    # Ajuste por etapa de diabetes
    stage_multiplier = 1.0
    if stage == 2:  # Ya tiene diabetes
        return "< 6 meses (Diabetes establecida - Manejo activo)"
    elif stage == 1:  # Prediabetes
        stage_multiplier = 0.8  # Acelera ligeramente

    # Calcular años estimados basado en datos epidemiológicos reales
    base_years = 0.0

    # Factor por glucosa (principal indicador) - rangos más conservadores
    if avg_glucose >= 200:  # Ya puede estar en diabetes
        base_years = 0.5
    elif avg_glucose >= 175:
        base_years = 1.5
    elif avg_glucose >= 150:  # Prediabetes fuerte
        base_years = 3.0
    elif avg_glucose >= 125:  # Prediabetes moderada
        base_years = 5.0
    elif avg_glucose >= 110:  # Prediabetes leve
        base_years = 7.0
    else:
        base_years = 10.0

    # Ajuste por edad (menos años a mayor edad) - más conservador
    age_adjustment = 1.0
    if age >= 65:
        age_adjustment = 0.7  # Más agresivo para ancianos
    elif age >= 55:
        age_adjustment = 0.8
    elif age >= 45:
        age_adjustment = 0.9
    elif age <= 12:  # Niños con prediabetes
        age_adjustment = 0.6  # Desarrollo más rápido en niños

    # Ajuste por factores cardiovasculares
    cardio_adjustment = 1.0
    if ldl is not None:
        if ldl > 160:
            cardio_adjustment *= 0.8  # Acelera moderadamente
        elif ldl > 130:
            cardio_adjustment *= 0.9

    if hdl is not None:
        if hdl < 40:
            cardio_adjustment *= 0.85  # Factor de riesgo importante
        elif hdl < 50:
            cardio_adjustment *= 0.95

    # Calcular años finales
    years = base_years * age_adjustment * cardio_adjustment * stage_multiplier

    # Aplicar factor de riesgo familiar (mayor factor = menos años)
    if family_risk_factor > 1.0:
        # Reducir años proporcionalmente al riesgo familiar
        reduction = (family_risk_factor - 1.0) * 0.3  # Máximo 30% de reducción por factor familiar
        years *= (1.0 - reduction)

    # Ajuste final por probabilidad - más granular
    if prob >= 0.8:
        years = min(years, 0.5)
        clasificacion = "Riesgo Crítico - Control inmediato"
    elif prob >= 0.7:
        years = min(years, 1.0)
        clasificacion = "Riesgo Muy Alto - Control urgente"
    elif prob >= 0.6:
        years = min(years, 2.0)
        clasificacion = "Riesgo Alto - Intervención inmediata"
    elif prob >= 0.5:
        years = min(years, 3.0)
        clasificacion = "Riesgo Alto-Moderado - Seguimiento cercano"
    elif prob >= 0.4:
        years = min(years, 5.0)
        clasificacion = "Riesgo Moderado - Seguimiento frecuente"
    elif prob >= 0.3:
        years = min(years, 7.0)
        clasificacion = "Riesgo Moderado-Bajo - Vigilancia regular"
    elif prob >= 0.2:
        years = min(years, 10.0)
        clasificacion = "Riesgo Bajo - Chequeos anuales"
    else:
        years = min(years, 15.0)
        clasificacion = "Riesgo Mínimo - Mantener hábitos saludables"

    # Rangos más precisos y realistas
    if years < 0.75:
        timeframe = "< 1 año"
    elif years < 1.5:
        timeframe = "1-2 años"
    elif years < 2.5:
        timeframe = "2-3 años"
    elif years < 3.5:
        timeframe = "3-4 años"
    elif years < 5:
        timeframe = "4-6 años"
    elif years < 7:
        timeframe = "6-8 años"
    elif years < 10:
        timeframe = "8-12 años"
    else:
        timeframe = "> 12 años"

    return f"{timeframe} ({clasificacion})"


def classify_lipid_profile(ldl, hdl, triglycerides, total_chol):
    """
    Clasifica el perfil lipídico con mayor precisión.
    Retorna: (riesgo_cardiovascular, descripción, score_riesgo)
    """
    riesgo = "Optimo"
    detalles = []
    risk_score = 0
    
    # Evaluación LDL (el malo) - más granular
    if ldl < 70:
        detalles.append("[OK] LDL (malo): Óptimo (muy protector)")
        risk_score += 0
    elif ldl < 100:
        detalles.append("[OK] LDL (malo): Óptimo")
        risk_score += 0
    elif ldl < 130:
        detalles.append("[!] LDL (malo): Deseable")
        riesgo = "Moderado"
        risk_score += 1
    elif ldl < 160:
        detalles.append("[!] LDL (malo): Elevado")
        riesgo = "Alto"
        risk_score += 2
    else:
        detalles.append("[XX] LDL (malo): Muy elevado")
        riesgo = "Critico"
        risk_score += 3
    
    # Evaluación HDL (el bueno) - más granular
    if hdl >= 60:
        detalles.append("[OK] HDL (bueno): Óptimo (protector fuerte)")
        risk_score -= 1  # Reduce riesgo
    elif hdl >= 50:
        detalles.append("[OK] HDL (bueno): Óptimo (protector)")
        risk_score -= 0.5
    elif hdl >= 40:
        detalles.append("[!] HDL (bueno): Deseable")
        risk_score += 0.5
    else:
        detalles.append("[XX] HDL (bueno): Bajo (factor de riesgo)")
        riesgo = "Alto" if riesgo == "Moderado" else riesgo
        risk_score += 2
    
    # Evaluación Triglicéridos - más granular
    if triglycerides < 100:
        detalles.append("[OK] Triglicéridos: Óptimos")
        risk_score -= 0.5
    elif triglycerides < 150:
        detalles.append("[OK] Triglicéridos: Óptimos")
        risk_score += 0
    elif triglycerides < 200:
        detalles.append("[!] Triglicéridos: Elevados")
        riesgo = "Alto" if riesgo == "Moderado" else riesgo
        risk_score += 1
    elif triglycerides < 500:
        detalles.append("[XX] Triglicéridos: Muy elevados")
        riesgo = "Critico"
        risk_score += 2
    else:
        detalles.append("[XX] Triglicéridos: Extremadamente elevados")
        riesgo = "Critico"
        risk_score += 3
    
    # Evaluación Colesterol Total
    if total_chol < 150:
        detalles.append("[OK] Colesterol Total: Muy deseable")
        risk_score -= 0.5
    elif total_chol < 200:
        detalles.append("[OK] Colesterol Total: Deseable")
        risk_score += 0
    elif total_chol < 240:
        detalles.append("[!] Colesterol Total: Elevado")
        riesgo = "Alto" if riesgo == "Moderado" else riesgo
        risk_score += 1
    else:
        detalles.append("[XX] Colesterol Total: Muy elevado")
        riesgo = "Critico"
        risk_score += 2
    
    # Normalizar score de riesgo
    normalized_risk = max(0, min(10, risk_score))
    
    return riesgo, detalles, normalized_risk


def generate_lipid_recommendations(ldl, hdl, triglycerides, risk_score=None):
    """Genera recomendaciones más específicas basadas en perfil lipídico."""
    recommendations = []
    
    # Recomendaciones por LDL
    if ldl > 190:
        recommendations.append("• LDL crítico: Consulta inmediata con cardiólogo")
        recommendations.append("• Posible inicio de estatinas de alta potencia")
    elif ldl > 160:
        recommendations.append("• Reducir grasas saturadas y trans drásticamente")
        recommendations.append("• Aumentar fibra soluble (avena, manzana, legumbres)")
        recommendations.append("• Considerar medicación si no mejora en 3 meses")
    elif ldl > 130:
        recommendations.append("• Moderar consumo de grasas saturadas")
        recommendations.append("• Incluir alimentos ricos en omega-3 (pescado, nueces)")
    
    # Recomendaciones por HDL
    if hdl < 35:
        recommendations.append("• HDL muy bajo: Aumentar ejercicio cardiovascular intensamente")
        recommendations.append("• Pérdida de peso significativa si IMC > 30")
        recommendations.append("• Posible suplementación con niacina (bajo supervisión)")
    elif hdl < 50:
        recommendations.append("• Aumentar actividad cardiovascular a 200 min/semana")
        recommendations.append("• Reducir peso si es necesario")
    
    # Recomendaciones por Triglicéridos
    if triglycerides > 500:
        recommendations.append("• Triglicéridos críticos: Evaluación médica urgente")
        recommendations.append("• Posible pancreatitis de riesgo")
    elif triglycerides > 200:
        recommendations.append("• Reducir azúcares refinados y alcohol completamente")
        recommendations.append("• Limitar carbohidratos refinados")
        recommendations.append("• Aumentar ejercicio aeróbico (45 min/día)")
    
    # Recomendaciones generales por score de riesgo
    if risk_score and risk_score > 5:
        recommendations.append("• Perfil lipídico de alto riesgo cardiovascular")
        recommendations.append("• Seguimiento cardiológico cada 3-6 meses")
    elif risk_score and risk_score > 3:
        recommendations.append("• Perfil lipídico moderado: Mejorar con cambios en estilo de vida")
    
    if not recommendations:
        recommendations.append("• Mantener los hábitos saludables actuales")
        recommendations.append("• Realizar chequeos lipídicos cada 6-12 meses")
    
    return recommendations

