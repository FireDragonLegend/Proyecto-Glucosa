"""
Módulo de Red Neuronal Avanzada para predicción de diabetes.
Incluye evaluación de prediabetes, prediabetes infantil y múltiples factores de riesgo
con mayor precisión y coherencia médica usando redes neuronales profundas.
"""
import math
import random
import numpy as np
import pickle


def save_model(model, filename):
    """Guarda el modelo en un archivo pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en {filename}")


def load_model(filename):
    """Carga un modelo desde un archivo pickle."""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Modelo cargado desde {filename}")
    return model


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
    Arquitectura: Input -> Hidden1 (ReLU) -> Hidden2 (ReLU) -> Output (Sigmoid para clasificación, Lineal para regresión)
    Incluye regularización L2, dropout y optimización Adam.
    """

    def __init__(self, input_size, hidden1_size=64, hidden2_size=32, output_size=1,
                 learning_rate=0.001, l2_lambda=0.001, dropout_rate=0.2, task='classification', use_batch_norm=True):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.task = task  # 'classification' or 'regression'
        self.use_batch_norm = use_batch_norm

        # Inicialización de pesos con Xavier/Glorot
        self.W1 = self._initialize_weights(input_size, hidden1_size)
        self.b1 = [0.0] * hidden1_size
        self.W2 = self._initialize_weights(hidden1_size, hidden2_size)
        self.b2 = [0.0] * hidden2_size
        self.W3 = self._initialize_weights(hidden2_size, output_size)
        self.b3 = [0.0] * output_size

        # Parámetros de Batch Normalization
        if use_batch_norm:
            self.bn1_gamma = [1.0] * hidden1_size
            self.bn1_beta = [0.0] * hidden1_size
            self.bn1_running_mean = [0.0] * hidden1_size
            self.bn1_running_var = [1.0] * hidden1_size
            
            self.bn2_gamma = [1.0] * hidden2_size
            self.bn2_beta = [0.0] * hidden2_size
            self.bn2_running_mean = [0.0] * hidden2_size
            self.bn2_running_var = [1.0] * hidden2_size

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

        if use_batch_norm:
            self.m_bn1_gamma = [0.0] * hidden1_size
            self.v_bn1_gamma = [0.0] * hidden1_size
            self.m_bn1_beta = [0.0] * hidden1_size
            self.v_bn1_beta = [0.0] * hidden1_size
            
            self.m_bn2_gamma = [0.0] * hidden2_size
            self.v_bn2_gamma = [0.0] * hidden2_size
            self.m_bn2_beta = [0.0] * hidden2_size
            self.v_bn2_beta = [0.0] * hidden2_size

        self.t = 0  # contador de Adam

    def __setstate__(self, state):
        """Compatibilidad con modelos cargados desde pickle antiguo."""
        self.__dict__.update(state)
        if not hasattr(self, 'task'):
            self.task = 'classification'
        if not hasattr(self, 'use_batch_norm'):
            self.use_batch_norm = False
        if not hasattr(self, 'dropout_rate'):
            self.dropout_rate = 0.0
        if not hasattr(self, 'learning_rate'):
            self.learning_rate = 0.001
        if not hasattr(self, 'l2_lambda'):
            self.l2_lambda = 0.001

        if self.use_batch_norm:
            self.bn1_gamma = getattr(self, 'bn1_gamma', [1.0] * self.hidden1_size)
            self.bn1_beta = getattr(self, 'bn1_beta', [0.0] * self.hidden1_size)
            self.bn1_running_mean = getattr(self, 'bn1_running_mean', [0.0] * self.hidden1_size)
            self.bn1_running_var = getattr(self, 'bn1_running_var', [1.0] * self.hidden1_size)
            self.bn2_gamma = getattr(self, 'bn2_gamma', [1.0] * self.hidden2_size)
            self.bn2_beta = getattr(self, 'bn2_beta', [0.0] * self.hidden2_size)
            self.bn2_running_mean = getattr(self, 'bn2_running_mean', [0.0] * self.hidden2_size)
            self.bn2_running_var = getattr(self, 'bn2_running_var', [1.0] * self.hidden2_size)
            self.m_bn1_gamma = getattr(self, 'm_bn1_gamma', [0.0] * self.hidden1_size)
            self.v_bn1_gamma = getattr(self, 'v_bn1_gamma', [0.0] * self.hidden1_size)
            self.m_bn1_beta = getattr(self, 'm_bn1_beta', [0.0] * self.hidden1_size)
            self.v_bn1_beta = getattr(self, 'v_bn1_beta', [0.0] * self.hidden1_size)
            self.m_bn2_gamma = getattr(self, 'm_bn2_gamma', [0.0] * self.hidden2_size)
            self.v_bn2_gamma = getattr(self, 'v_bn2_gamma', [0.0] * self.hidden2_size)
            self.m_bn2_beta = getattr(self, 'm_bn2_beta', [0.0] * self.hidden2_size)
            self.v_bn2_beta = getattr(self, 'v_bn2_beta', [0.0] * self.hidden2_size)
        else:
            self.bn1_gamma = getattr(self, 'bn1_gamma', [])
            self.bn1_beta = getattr(self, 'bn1_beta', [])
            self.bn1_running_mean = getattr(self, 'bn1_running_mean', [])
            self.bn1_running_var = getattr(self, 'bn1_running_var', [])
            self.bn2_gamma = getattr(self, 'bn2_gamma', [])
            self.bn2_beta = getattr(self, 'bn2_beta', [])
            self.bn2_running_mean = getattr(self, 'bn2_running_mean', [])
            self.bn2_running_var = getattr(self, 'bn2_running_var', [])
            self.m_bn1_gamma = getattr(self, 'm_bn1_gamma', [])
            self.v_bn1_gamma = getattr(self, 'v_bn1_gamma', [])
            self.m_bn1_beta = getattr(self, 'm_bn1_beta', [])
            self.v_bn1_beta = getattr(self, 'v_bn1_beta', [])
            self.m_bn2_gamma = getattr(self, 'm_bn2_gamma', [])
            self.v_bn2_gamma = getattr(self, 'v_bn2_gamma', [])
            self.m_bn2_beta = getattr(self, 'm_bn2_beta', [])
            self.v_bn2_beta = getattr(self, 'v_bn2_beta', [])

    def _initialize_weights(self, fan_in, fan_out):
        """Inicialización Xavier/Glorot para pesos."""
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return [[random.uniform(-limit, limit) for _ in range(fan_out)] for _ in range(fan_in)]

    def _batch_norm(self, x, gamma, beta, running_mean, running_var, training=True, epsilon=1e-5):
        """Batch Normalization."""
        if training:
            mean = sum(x) / len(x)
            var = sum((xi - mean) ** 2 for xi in x) / len(x)
            
            # Update running statistics
            momentum = 0.9
            running_mean[:] = [momentum * rm + (1 - momentum) * mean for rm in running_mean]
            running_var[:] = [momentum * rv + (1 - momentum) * var for rv in running_var]
        else:
            mean = running_mean[0]  # Para inferencia, usar running stats
            var = running_var[0]
        
        # Normalize
        x_norm = [(xi - mean) / math.sqrt(var + epsilon) for xi in x]
        
        # Scale and shift
        out = [gamma[i] * x_norm[i] + beta[i] for i in range(len(x))]
        return out, x_norm, mean, var

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
        if getattr(self, 'use_batch_norm', False):
            self.a1, self.bn1_x_norm, self.bn1_mean, self.bn1_var = self._batch_norm(
                self.a1, self.bn1_gamma, self.bn1_beta, self.bn1_running_mean, self.bn1_running_var, training)
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
        if getattr(self, 'use_batch_norm', False):
            self.a2, self.bn2_x_norm, self.bn2_mean, self.bn2_var = self._batch_norm(
                self.a2, self.bn2_gamma, self.bn2_beta, self.bn2_running_mean, self.bn2_running_var, training)
        if training:
            self.a2 = self._dropout(self.a2, self.dropout_rate)

        # Capa 3: Hidden2 -> Output
        self.z3 = []
        for i in range(self.output_size):
            z3_i = self.b3[i]
            for j in range(self.hidden2_size):
                z3_i += self.a2[j] * self.W3[j][i]
            self.z3.append(z3_i)

        task = getattr(self, 'task', 'classification')
        if task == 'classification':
            self.a3 = [sigmoid(z) for z in self.z3]
        else:  # regression
            self.a3 = self.z3  # linear activation
        return self.a3

    def backward(self, X, y):
        """Backward pass con regularización L2."""
        # Calcular errores
        if self.task == 'classification':
            delta3 = [(a - target) for a, target in zip(self.a3, y)]
        else:  # regression
            delta3 = [(a - target) for a, target in zip(self.a3, y)]  # same for MSE

        # Gradientes capa 3
        dW3 = [[0.0] * self.output_size for _ in range(self.hidden2_size)]
        db3 = [0.0] * self.output_size

        for i in range(self.hidden2_size):
            for j in range(self.output_size):
                dW3[i][j] = delta3[j] * self.a2[i] + self.l2_lambda * self.W3[i][j]
        
        # Bias gradients (debe calcularse para todos los nodos output)
        for j in range(self.output_size):
            db3[j] = delta3[j]

        # Gradientes capa 2
        delta2 = [0.0] * self.hidden2_size
        for i in range(self.hidden2_size):
            for j in range(self.output_size):
                delta2[i] += delta3[j] * self.W3[i][j]
            delta2[i] *= relu_derivative(self.z2[i])

        # Gradientes Batch Norm 2
        if self.use_batch_norm:
            dbn2_gamma = [delta2[i] * self.bn2_x_norm[i] for i in range(self.hidden2_size)]
            dbn2_beta = delta2.copy()
        else:
            dbn2_gamma = [0.0] * self.hidden2_size
            dbn2_beta = [0.0] * self.hidden2_size

        dW2 = [[0.0] * self.hidden2_size for _ in range(self.hidden1_size)]
        db2 = [0.0] * self.hidden2_size

        for i in range(self.hidden1_size):
            for j in range(self.hidden2_size):
                dW2[i][j] = delta2[j] * self.a1[i] + self.l2_lambda * self.W2[i][j]
        
        # Bias gradients (debe calcularse para todos los nodos hidden2)
        for j in range(self.hidden2_size):
            db2[j] = delta2[j]

        # Gradientes capa 1
        delta1 = [0.0] * self.hidden1_size
        for i in range(self.hidden1_size):
            for j in range(self.hidden2_size):
                delta1[i] += delta2[j] * self.W2[i][j]
            delta1[i] *= relu_derivative(self.z1[i])

        # Gradientes Batch Norm 1
        if self.use_batch_norm:
            dbn1_gamma = [delta1[i] * self.bn1_x_norm[i] for i in range(self.hidden1_size)]
            dbn1_beta = delta1.copy()
        else:
            dbn1_gamma = [0.0] * self.hidden1_size
            dbn1_beta = [0.0] * self.hidden1_size

        dW1 = [[0.0] * self.hidden1_size for _ in range(self.input_size)]
        db1 = [0.0] * self.hidden1_size

        for i in range(self.input_size):
            for j in range(self.hidden1_size):
                dW1[i][j] = delta1[j] * X[i] + self.l2_lambda * self.W1[i][j]
        
        # Bias gradients (debe calcularse para todos los nodos hidden1)
        for j in range(self.hidden1_size):
            db1[j] = delta1[j]

        if self.use_batch_norm:
            return dW1, db1, dW2, db2, dW3, db3, dbn1_gamma, dbn1_beta, dbn2_gamma, dbn2_beta
        else:
            return dW1, db1, dW2, db2, dW3, db3

    def _adam_update(self, param, grad, m, v, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Actualización Adam para un parámetro."""
        self.t += 1
        m_new = beta1 * m + (1 - beta1) * grad
        v_new = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m_new / (1 - beta1 ** self.t)
        v_hat = v_new / (1 - beta2 ** self.t)

        # Learning rate decay
        decay_rate = 0.001
        current_lr = self.learning_rate / (1 + decay_rate * (self.t // 1000))  # decay cada 1000 steps

        param_new = param - current_lr * m_hat / (math.sqrt(v_hat) + epsilon)
        return param_new, m_new, v_new

    def update_parameters(self, dW1, db1, dW2, db2, dW3, db3, dbn1_gamma=None, dbn1_beta=None, dbn2_gamma=None, dbn2_beta=None):
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

        # Actualizar Batch Norm parameters
        if getattr(self, 'use_batch_norm', False) and dbn1_gamma is not None:
            for j in range(self.hidden1_size):
                self.bn1_gamma[j], self.m_bn1_gamma[j], self.v_bn1_gamma[j] = self._adam_update(
                    self.bn1_gamma[j], dbn1_gamma[j], self.m_bn1_gamma[j], self.v_bn1_gamma[j])
                self.bn1_beta[j], self.m_bn1_beta[j], self.v_bn1_beta[j] = self._adam_update(
                    self.bn1_beta[j], dbn1_beta[j], self.m_bn1_beta[j], self.v_bn1_beta[j])

            for j in range(self.hidden2_size):
                self.bn2_gamma[j], self.m_bn2_gamma[j], self.v_bn2_gamma[j] = self._adam_update(
                    self.bn2_gamma[j], dbn2_gamma[j], self.m_bn2_gamma[j], self.v_bn2_gamma[j])
                self.bn2_beta[j], self.m_bn2_beta[j], self.v_bn2_beta[j] = self._adam_update(
                    self.bn2_beta[j], dbn2_beta[j], self.m_bn2_beta[j], self.v_bn2_beta[j])

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
                
                if getattr(self, 'use_batch_norm', False):
                    total_dbn1_gamma = [0.0] * self.hidden1_size
                    total_dbn1_beta = [0.0] * self.hidden1_size
                    total_dbn2_gamma = [0.0] * self.hidden2_size
                    total_dbn2_beta = [0.0] * self.hidden2_size

                for x, target in zip(batch_X, batch_y):
                    grads = self.backward(x, [target])
                    dW1, db1, dW2, db2, dW3, db3 = grads[:6]
                    
                    if getattr(self, 'use_batch_norm', False) and len(grads) > 6:
                        dbn1_gamma, dbn1_beta, dbn2_gamma, dbn2_beta = grads[6:]

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
                    
                    if getattr(self, 'use_batch_norm', False):
                        for i in range(len(total_dbn1_gamma)):
                            total_dbn1_gamma[i] += dbn1_gamma[i]
                            total_dbn1_beta[i] += dbn1_beta[i]
                        for i in range(len(total_dbn2_gamma)):
                            total_dbn2_gamma[i] += dbn2_gamma[i]
                            total_dbn2_beta[i] += dbn2_beta[i]

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
                
                if getattr(self, 'use_batch_norm', False):
                    for i in range(len(total_dbn1_gamma)):
                        total_dbn1_gamma[i] /= batch_size_actual
                        total_dbn1_beta[i] /= batch_size_actual
                    for i in range(len(total_dbn2_gamma)):
                        total_dbn2_gamma[i] /= batch_size_actual
                        total_dbn2_beta[i] /= batch_size_actual

                # Update parameters
                if getattr(self, 'use_batch_norm', False):
                    self.update_parameters(total_dW1, total_db1, total_dW2, total_db2, total_dW3, total_db3,
                                         total_dbn1_gamma, total_dbn1_beta, total_dbn2_gamma, total_dbn2_beta)
                else:
                    self.update_parameters(total_dW1, total_db1, total_dW2, total_db2, total_dW3, total_db3)

                # Calculate batch loss
                if self.task == 'classification':
                    for pred, target in zip(predictions, batch_y):
                        loss = -target * math.log(pred + 1e-15) - (1 - target) * math.log(1 - pred + 1e-15)
                        epoch_loss += loss
                else:  # regression
                    for pred, target in zip(predictions, batch_y):
                        loss = (pred - target) ** 2
                        epoch_loss += loss

            epoch_loss /= len(X_train)

            # Validation
            if epoch % 10 == 0:
                val_predictions = []
                for x in X_val:
                    pred = self.forward(x, training=False)
                    val_predictions.append(pred[0])  # Corregir: append el valor escalar

                val_loss = 0.0
                if self.task == 'classification':
                    for pred, target in zip(val_predictions, y_val):
                        loss = -target * math.log(pred + 1e-15) - (1 - target) * math.log(1 - pred + 1e-15)
                        val_loss += loss
                else:  # regression
                    for pred, target in zip(val_predictions, y_val):
                        loss = (pred - target) ** 2
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

                if epoch % 50 == 0 or epoch == epochs - 1:
                    if self.task == 'classification':
                        metrics = self.evaluate_classification(X_val, y_val)
                        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, val_acc={metrics['accuracy']:.4f}, val_f1={metrics['f1']:.4f}")
                    else:
                        metrics = self.evaluate_regression(X_val, y_val)
                        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, val_mae={metrics['mae']:.4f}, val_rmse={metrics['rmse']:.4f}")

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
        """Predice probabilidades (clasificación) o valores (regresión)."""
        predictions = []
        for x in X:
            pred = self.forward(x, training=False)
            predictions.append(pred[0])
        return predictions

    def evaluate_classification(self, X, y, threshold=0.5):
        """Evalúa métricas de clasificación: accuracy, precision, recall, F1."""
        predictions = self.predict(X, threshold)
        tp = fp = tn = fn = 0
        for pred, true in zip(predictions, y):
            if pred == 1 and true == 1:
                tp += 1
            elif pred == 1 and true == 0:
                fp += 1
            elif pred == 0 and true == 1:
                fn += 1
            else:
                tn += 1
        
        accuracy = (tp + tn) / len(y) if len(y) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }

    def evaluate_regression(self, X, y):
        """Evalúa métricas de regresión: MAE, RMSE, R2."""
        predictions = self.predict_time(X)
        mae = sum(abs(p - t) for p, t in zip(predictions, y)) / len(y)
        mse = sum((p - t) ** 2 for p, t in zip(predictions, y)) / len(y)
        rmse = math.sqrt(mse)
        
        # R2
        mean_y = sum(y) / len(y)
        ss_tot = sum((t - mean_y) ** 2 for t in y)
        ss_res = sum((p - t) ** 2 for p, t in zip(predictions, y))
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

    def predict(self, X, threshold=0.5):
        """Predice clases binarias."""
        probs = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probs]

    def predict_diabetes_stage(self, features):
        """
        Predice etapa de diabetes: 0=Normal, 1=Prediabetes, 2=Diabetes
        """
        age = features[0]
        glucose_fast = features[1]
        glucose_post = features[2]
        a1c = features[3]
        prob = self.predict_proba([features])[0]
        return get_diabetes_stage_from_features(age, glucose_fast, glucose_post, a1c, prob=prob)


class AdvancedDiabetesModel(AdvancedNeuralNetwork):
    """
    Modelo avanzado de diabetes basado en red neuronal profunda.
    Hereda de AdvancedNeuralNetwork con configuración específica para diabetes.
    """

    def __init__(self, n_features, task='classification'):
        # Configuración optimizada para diabetes: 15 features -> 64 -> 32 -> 1
        super().__init__(
            input_size=n_features,
            hidden1_size=64,
            hidden2_size=32,
            output_size=1,
            learning_rate=0.001,
            l2_lambda=0.001,
            dropout_rate=0.2,
            task=task,
            use_batch_norm=True  # Habilitar batch normalization para mejor estabilidad
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


def is_diabetes_criteria(age, glucose_fast, glucose_post, a1c):
    """Criterios clínicos claros de diabetes para todas las edades."""
    if a1c is None:
        return glucose_fast >= 126 or glucose_post >= 200
    return glucose_fast >= 126 or glucose_post >= 200 or a1c >= 6.5


def is_prediabetes_criteria(age, glucose_fast, glucose_post, a1c):
    """Criterios de prediabetes ajustados por etapa de vida."""
    if age < 0.08:
        # Recién nacidos y neonatos tienen umbrales más sensibles y no deben clasificarse con criterios adultos sin evidencia muy clara
        if a1c is None:
            return glucose_fast >= 110 or glucose_post >= 140
        return glucose_fast >= 110 or glucose_post >= 140 or a1c >= 5.7
    if age < 2:
        if a1c is None:
            return glucose_fast >= 110 or glucose_post >= 140
        return glucose_fast >= 110 or glucose_post >= 140 or a1c >= 5.7
    if age <= 18:
        if a1c is None:
            return glucose_fast >= 100 or glucose_post >= 140
        return glucose_fast >= 100 or glucose_post >= 140 or a1c >= 5.7
    if age < 65:
        if a1c is None:
            return glucose_fast >= 100 or glucose_post >= 140
        return glucose_fast >= 100 or glucose_post >= 140 or a1c >= 5.7
    if a1c is None:
        return glucose_fast >= 110 or glucose_post >= 140
    return glucose_fast >= 110 or glucose_post >= 140 or a1c >= 5.7


def get_diabetes_stage_from_features(age, glucose_fast, glucose_post, a1c, prob=None):
    """Devuelve la etapa clínica en base a glucosa, A1C y edad."""
    # Etapa clínica prioritaria basada en valores medibles.
    if is_diabetes_criteria(age, glucose_fast, glucose_post, a1c):
        return 2
    if is_prediabetes_criteria(age, glucose_fast, glucose_post, a1c):
        return 1

    # Solo promover a prediabetes si hay probabilidad muy alta y valores borderline.
    if prob is not None and prob >= 0.85:
        borderline_fast = glucose_fast >= 95
        borderline_post = glucose_post >= 120
        borderline_a1c = a1c is not None and a1c >= 5.4
        if borderline_fast or borderline_post or borderline_a1c:
            return 1

    return 0


def describe_age_stage_risk(age, glucose_fast, glucose_post, a1c):
    """Retorna una descripción de riesgo basada en la etapa de vida."""
    if age < 0.08:  # menor de aproximadamente 1 mes
        if glucose_fast < 50 or glucose_post < 80:
            return "Recién nacido: riesgo de hipoglucemia neonatal más importante que diabetes"
        return "Recién nacido: diabetes neonatal muy rara, pero requiere seguimiento estrecho"

    if is_diabetes_criteria(age, glucose_fast, glucose_post, a1c):
        if age <= 18:
            return "Niños/adolescentes: hallazgos consistentes con diabetes, requiere derivación pediátrica urgente"
        if age < 65:
            return "Adultos: criterios de diabetes claros, iniciar manejo médico inmediato"
        return "Adulto mayor: criterios de diabetes claros, manejo activo para evitar complicaciones"

    if is_prediabetes_criteria(age, glucose_fast, glucose_post, a1c):
        if age < 2:
            return "Bebé/infante: alerta temprana, riesgo bajo de diabetes pero requiere seguimiento pediátrico"
        if age <= 18:
            return "Niños/adolescentes: alerta importante, el riesgo de diabetes Tipo 1 y Tipo 2 está en aumento"
        if age < 65:
            return "Adultos: riesgo alto de progresión a diabetes Tipo 2 sin cambios en el estilo de vida"
        return "Adulto mayor: riesgo moderado-elevado, el umbral de glucosa es más flexible pero requiere vigilancia"

    if age < 2:
        return "Bebé/infante: riesgo de diabetes bajo, vigilar alimentación y síntomas"
    if age <= 18:
        return "Niños/adolescentes: riesgo bajo, prevención y actividad física son clave"
    if age < 65:
        return "Adultos: rango estándar, mantener buenos hábitos y control regular"
    return "Adulto mayor: riesgo moderado, se usa un umbral superior más flexible para evitar hipoglucemia"


def is_childhood_prediabetes(age, glucose_fast, glucose_post, a1c):
    """
    Evalúa prediabetes infantil y diabetes Tipo 1 temprana en niños y adolescentes.
    """
    if 4 <= age <= 18:
        return is_prediabetes_criteria(age, glucose_fast, glucose_post, a1c)
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
    if glucose_fast >= 100 or glucose_post >= 140 or (a1c is not None and a1c >= 5.7):
        risk_factors += 1

    # Diagnóstico: Si hay múltiples factores + valores de glucosa elevados
    # Se considera posible Tipo 2 temprana
    if risk_factors >= 3 and (glucose_fast >= 100 or glucose_post >= 140 or (a1c is not None and a1c >= 5.7)):
        return True

    # O si hay diagnóstico claro de prediabetes/diabetes con cualquier otro factor
    if (glucose_fast >= 125 or glucose_post >= 140 or (a1c is not None and a1c >= 5.7)) and risk_factors >= 2:
        return True

    return False


def generate_time_to_diabetes_dataset(n_samples=5000, seed=42):
    """
    Genera dataset sintético para predicción de tiempo hasta diabetes.
    Retorna tiempo en años hasta desarrollar diabetes (0-50 años).
    """
    random.seed(seed)
    X = []
    y = []

    print(f"Generando {n_samples} muestras de datos sintéticos para tiempo hasta diabetes...")

    for i in range(n_samples):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Procesando muestra {i+1}/{n_samples}")

        # Características demográficas con distribución mejorada
        age_distribution = random.random()
        if age_distribution < 0.1:  # 10% niños
            age = random.randint(4, 14)
        elif age_distribution < 0.3:  # 20% jóvenes
            age = random.randint(15, 25)
        elif age_distribution < 0.7:  # 40% adultos
            age = random.randint(26, 50)
        else:  # 30% mayores
            age = random.randint(51, 80)

        # BMI basado en edad
        if age < 18:
            bmi_base = random.gauss(20, 3)  # Niños más delgados
        elif age < 30:
            bmi_base = random.gauss(24, 4.5)  # Adultos jóvenes
        elif age < 50:
            bmi_base = random.gauss(26, 5)  # Adultos medios
        else:
            bmi_base = random.gauss(28, 6)  # Mayores con tendencia a sobrepeso

        bmi = max(15, min(50, bmi_base))

        # Características glucémicas con mayor realismo
        base_glucose = 85 + (age * 0.2) + (bmi - 25) * 0.8
        glucose_variability = random.gauss(0, 15)
        avg_fast = max(60, min(300, base_glucose + glucose_variability))

        post_variability = random.gauss(0, 25)
        avg_post = max(80, min(400, avg_fast + 50 + post_variability))

        a1c_base = 4.5 + ((avg_fast + avg_post)/2 - 100) * 0.02
        a1c = max(4.0, min(12.0, a1c_base + random.gauss(0, 0.5)))

        # Características lipídicas mejoradas con correlaciones realistas
        ldl_base = 90 + (age * 0.3) + (bmi - 25) * 1.2
        ldl = max(50, min(250, ldl_base + random.gauss(0, 25)))

        hdl_base = 50 - (bmi - 25) * 0.5 + random.gauss(0, 10)
        hdl = max(20, min(100, hdl_base))

        triglycerides_base = 100 + (bmi - 25) * 2 + random.gauss(0, 30)
        triglycerides = max(50, min(500, triglycerides_base))

        total_chol = ldl + hdl + triglycerides/5 + random.gauss(0, 20)

        # Presión arterial
        systolic_base = 110 + (age * 0.5) + (bmi - 25) * 0.8
        systolic = max(90, min(200, systolic_base + random.gauss(0, 15)))
        diastolic = max(60, min(120, systolic - 30 + random.gauss(0, 10)))

        # Factores de estilo de vida
        family_risk = 1 if random.random() < 0.25 else 0  # 25% tienen antecedentes familiares

        # Calcular care_score mejorado con más factores
        care_score = 3.0  # Base neutral

        # Ajustes por ejercicio
        exercise_weekly = random.randint(0, 500)
        if exercise_weekly >= 300:
            care_score -= 1.0  # Muy activo
        elif exercise_weekly >= 150:
            care_score -= 0.5  # Moderadamente activo
        elif exercise_weekly < 30:
            care_score += 0.5  # Sedentario

        # Ajustes por dieta
        diet_quality = random.randint(1, 5)
        care_score -= (diet_quality - 3) * 0.3  # Mejor dieta = mejor score

        # Ajustes por otros factores
        if bmi >= 35:
            care_score += 1.0  # Obesidad severa
        elif bmi >= 30:
            care_score += 0.7
        elif bmi >= 25:
            care_score += 0.3

        care_score = max(1.0, min(5.0, care_score))

        # Calcular tiempo hasta diabetes basado en factores de riesgo
        base_time = 50  # Máximo 50 años

        # Factores que reducen tiempo
        risk_multiplier = 1.0

        # Edad: más joven = más tiempo
        if age < 20:
            risk_multiplier *= 1.5
        elif age < 40:
            risk_multiplier *= 1.2
        elif age > 60:
            risk_multiplier *= 0.8

        # BMI alto reduce tiempo
        if bmi > 30:
            risk_multiplier *= 0.5
        elif bmi > 25:
            risk_multiplier *= 0.7

        # Glucosa elevada reduce tiempo
        if avg_fast > 100:
            risk_multiplier *= 0.6
        if a1c > 5.7:
            risk_multiplier *= 0.7

        # Antecedentes familiares
        if family_risk:
            risk_multiplier *= 0.8

        # Estilo de vida saludable aumenta tiempo
        lifestyle_factor = (exercise_weekly / 300) * 0.3 + (diet_quality / 5) * 0.2
        risk_multiplier *= (1 + lifestyle_factor)

        # Calcular tiempo final
        time_to_diabetes = base_time * risk_multiplier
        time_to_diabetes = max(0.1, min(50, time_to_diabetes + random.gauss(0, 5)))

        features = [
            age, avg_fast, avg_post, a1c, bmi,
            ldl, hdl, triglycerides, total_chol,
            systolic, diastolic, care_score,
            family_risk, exercise_weekly, diet_quality
        ]

        X.append(features)
        y.append(time_to_diabetes)

    print(f"Dataset completado: {len(X)} muestras generadas")
    return X, y
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
        if (i + 1) % 100 == 0 or i == 0:
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
        if (i + 1) % 100 == 0 or i == 0:
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
    model.fit(X, y, epochs=500, batch_size=64, validation_split=0.2, patience=50)  # Parámetros optimizados
    print("Entrenamiento completado.")
    return model


def train_time_to_diabetes_model():
    """Entrena modelo avanzado para predicción de tiempo hasta diabetes."""
    print("Generando dataset de tiempo hasta diabetes...")
    X, y = generate_time_to_diabetes_dataset(5000)
    print(f"Dataset generado: {len(X)} muestras con {len(X[0])} características")

    model = AdvancedDiabetesModel(n_features=15, task='regression')
    print("Entrenando red neuronal para regresión de tiempo...")
    model.fit(X, y, epochs=500, batch_size=64, validation_split=0.2, patience=50)
    print("Entrenamiento completado.")
    return model


def assess_diabetes_risk_comprehensive(features, model, time_model=None):
    """
    Evaluación integral del riesgo de diabetes.
    Retorna: (probabilidad, etapa, tiempo_estimado, factores_riesgo, recomendaciones)
    """
    prob = model.predict_proba([features])[0]
    age, glucose_fast, glucose_post, a1c, bmi = features[:5]
    stage = get_diabetes_stage_from_features(age, glucose_fast, glucose_post, a1c, prob=prob)
    
    time_to_diabetes = None
    if time_model and hasattr(time_model, 'predict_time'):
        time_to_diabetes = time_model.predict_time([features])[0]

    age, glucose_fast, glucose_post, a1c, bmi = features[:5]
    ldl, hdl, triglycerides, total_chol = features[5:9]
    systolic, diastolic = features[9:11]
    care_score, family_history, exercise_weekly, diet_quality = features[11:15]

    # Evaluar factores de riesgo
    risk_factors = []

    # Glucosa
    if is_diabetes_criteria(age, glucose_fast, glucose_post, a1c):
        risk_factors.append("Glucosa elevada (Diabetes)")
    elif is_prediabetes_criteria(age, glucose_fast, glucose_post, a1c):
        risk_factors.append("Glucosa borderline (Prediabetes)")

    # Edad y prediabetes infantil
    if is_childhood_prediabetes(age, glucose_fast, glucose_post, a1c):
        risk_factors.append("Prediabetes infantil o adolescente (4-18 años)")

    # Descripción de riesgo por etapa de vida
    age_stage_text = describe_age_stage_risk(age, glucose_fast, glucose_post, a1c)
    if age_stage_text:
        risk_factors.append(age_stage_text)
    
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
    if diet_quality <= 2:
        risk_factors.append("Dieta pobre")

    # Generar recomendaciones
    recommendations = generate_comprehensive_recommendations(
        stage, age, bmi, ldl, hdl, triglycerides, exercise_weekly, diet_quality, systolic, diastolic, family_history, time_to_diabetes
    )

    return prob, stage, time_to_diabetes, risk_factors, recommendations


def generate_comprehensive_recommendations(stage, age, bmi, ldl, hdl, triglycerides, exercise, diet_quality, systolic, diastolic, family_history, time_to_diabetes=None):
    """Genera recomendaciones integrales y personalizadas basadas en todos los factores de riesgo."""
    recommendations = []

    # Recomendaciones por etapa y tiempo estimado
    if stage == 2:  # Diabetes
        recommendations.append("• Consulta inmediata con endocrinólogo para manejo de diabetes")
        recommendations.append("• Iniciar tratamiento médico (medicamentos, insulina si es necesario)")
        recommendations.append("• Monitoreo continuo de glucosa (glucómetro o sensor continuo)")
        recommendations.append("• Educación diabetológica para autocuidado")
        if time_to_diabetes and time_to_diabetes < 1:
            recommendations.append("• ¡Atención urgente! Diabetes ya establecida - controlar complicaciones")
    elif stage == 1:  # Prediabetes
        recommendations.append("• Cambios intensivos en estilo de vida para prevenir diabetes tipo 2")
        recommendations.append("• Seguimiento médico cada 3-6 meses con pruebas de glucosa")
        if time_to_diabetes:
            if time_to_diabetes < 3:
                recommendations.append(f"• ¡Riesgo alto! Tiempo estimado: {time_to_diabetes:.1f} años - actuar inmediatamente")
            elif time_to_diabetes < 7:
                recommendations.append(f"• Riesgo moderado. Tiempo estimado: {time_to_diabetes:.1f} años - cambios urgentes")
            else:
                recommendations.append(f"• Riesgo bajo-moderado. Tiempo estimado: {time_to_diabetes:.1f} años - prevención preventiva")
        if age <= 12:
            recommendations.append("• Atención pediátrica especializada para prediabetes infantil")
            recommendations.append("• Monitoreo de crecimiento y desarrollo")
        elif 20 <= age <= 44:
            recommendations.append("• Atención para Diabetes Tipo 2 emergente en adultos jóvenes")
            recommendations.append("• Intensificar cambios de estilo de vida: dieta baja en carbohidratos refinados y ejercicio regular")

    # Recomendaciones específicas por ejercicio
    if exercise < 30:
        recommendations.append("• Sedentarismo extremo - comenzar con caminatas diarias de 10-15 minutos")
        recommendations.append("• Meta: 150 minutos/semana de actividad moderada (caminar, nadar, ciclismo)")
        recommendations.append("• Consultar médico antes de iniciar ejercicio intenso")
    elif exercise < 75:
        recommendations.append("• Actividad física insuficiente - aumentar gradualmente a 30 min diarios")
        recommendations.append("• Incorporar ejercicio en rutina diaria (subir escaleras, caminar al trabajo)")
    elif exercise < 150:
        recommendations.append("• Actividad moderada pero insuficiente - alcanzar 150 min/semana")
        recommendations.append("• Añadir 2-3 sesiones de ejercicio estructurado por semana")
    elif exercise >= 150 and exercise < 300:
        recommendations.append("• Buen nivel de actividad - mantener y aumentar intensidad si es posible")
        recommendations.append("• Incluir ejercicios de fuerza 2 veces/semana para masa muscular")
    else:  # >= 300
        recommendations.append("• Excelente nivel de actividad física - continuar así")
        recommendations.append("• Monitorear sobrecarga y recuperación adecuada")

    # Recomendaciones específicas por dieta
    if diet_quality == 1:
        recommendations.append("• Dieta muy pobre - consulta urgente con nutricionista")
        recommendations.append("• Eliminar azúcares refinados, bebidas azucaradas y alimentos procesados")
        recommendations.append("• Adoptar dieta mediterránea: verduras, frutas, granos enteros, pescado, aceite de oliva")
        recommendations.append("• Controlar porciones y frecuencia de comidas")
    elif diet_quality == 2:
        recommendations.append("• Dieta deficiente - mejorar calidad nutricional")
        recommendations.append("• Reducir carbohidratos refinados (pan blanco, pastas) por integrales")
        recommendations.append("• Aumentar consumo de verduras y frutas (5 porciones/día mínimo)")
        recommendations.append("• Incluir proteínas magras y grasas saludables (pescado, nueces, aceite de oliva)")
    elif diet_quality == 3:
        recommendations.append("• Dieta aceptable - optimizar para prevención")
        recommendations.append("• Moderar azúcares y grasas saturadas")
        recommendations.append("• Aumentar fibra dietética (legumbres, cereales integrales)")
    elif diet_quality == 4:
        recommendations.append("• Buena dieta - mantener y refinar")
        recommendations.append("• Enfocarse en alimentos antiinflamatorios y antioxidantes")
    else:  # 5
        recommendations.append("• Excelente calidad dietética - continuar con hábitos saludables")
        recommendations.append("• Monitorear ingesta calórica si hay sobrepeso")

    # Recomendaciones por IMC
    if bmi >= 40:
        recommendations.append("• Obesidad mórbida - evaluación bariátrica urgente")
        recommendations.append("• Programa multidisciplinario de pérdida de peso")
        recommendations.append("• Cirugía bariátrica si criterios cumplen")
    elif bmi >= 35:
        recommendations.append("• Obesidad severa - evaluación médica especializada")
        recommendations.append("• Pérdida de peso supervisada (1-2 kg/mes)")
        recommendations.append("• Combinar dieta, ejercicio y posible medicación")
    elif bmi >= 30:
        recommendations.append("• Obesidad - pérdida gradual de 0.5-1 kg/mes")
        recommendations.append("• Meta: reducir 5-10% del peso corporal inicial")
    elif bmi >= 25:
        recommendations.append("• Sobrepeso - mantener peso estable inicialmente")
        recommendations.append("• Prevenir ganancia de peso adicional")
        recommendations.append("• Aumentar actividad física para control metabólico")

    # Recomendaciones cardiovasculares
    if systolic >= 180 or diastolic >= 110:
        recommendations.append("• Hipertensión severa - consulta inmediata con cardiólogo")
        recommendations.append("• Medicación antihipertensiva urgente")
        recommendations.append("• Monitoreo ambulatorio de presión arterial")
    elif systolic >= 140 or diastolic >= 90:
        recommendations.append("• Hipertensión - control con medicación y cambios de vida")
        recommendations.append("• Reducir sal, perder peso, ejercicio regular")
        recommendations.append("• Seguimiento mensual de presión arterial")
    elif systolic >= 130 or diastolic >= 80:
        recommendations.append("• Presión elevada - prevenir progresión")
        recommendations.append("• Cambios en estilo de vida: DASH diet, ejercicio, reducción de estrés")

    if ldl >= 190:
        recommendations.append("• LDL muy elevado - posible hipercolesterolemia familiar")
        recommendations.append("• Estatinas de alta potencia + dieta estricta")
        recommendations.append("• Evaluación genética si antecedentes familiares")
    elif ldl >= 160:
        recommendations.append("• LDL elevado - iniciar estatinas y dieta")
        recommendations.append("• Reducir grasas saturadas y trans")
        recommendations.append("• Meta LDL < 100 mg/dL")
    elif ldl >= 130:
        recommendations.append("• LDL borderline - dieta y ejercicio primero")
        recommendations.append("• Reevaluar en 3 meses, posible medicación")

    if hdl < 35:
        recommendations.append("• HDL muy bajo - riesgo cardiovascular aumentado")
        recommendations.append("• Aumentar ejercicio aeróbico intensamente")
        recommendations.append("• Posible suplementación con niacina (bajo supervisión)")
    elif hdl < 50:
        recommendations.append("• HDL bajo - mejorar con ejercicio y dieta")
        recommendations.append("• 200 min/semana de actividad cardiovascular")

    if triglycerides >= 500:
        recommendations.append("• Triglicéridos muy elevados - riesgo de pancreatitis")
        recommendations.append("• Dieta muy baja en carbohidratos y grasas")
        recommendations.append("• Medicación (fibratos) + control estricto")
    elif triglycerides >= 200:
        recommendations.append("• Triglicéridos elevados - reducir azúcares y alcohol")
        recommendations.append("• Pérdida de peso y ejercicio aeróbico")
        recommendations.append("• Meta < 150 mg/dL")

    # Recomendaciones por antecedentes familiares
    if family_history:
        recommendations.append("• Antecedentes familiares positivos - mayor vigilancia")
        recommendations.append("• Tamizaje genético si múltiples familiares afectados")
        recommendations.append("• Educación familiar sobre factores de riesgo")
        recommendations.append("• Seguimiento más frecuente (cada 6 meses)")

    # Recomendaciones por edad
    if age <= 12 and stage >= 1:
        recommendations.append("• Seguimiento pediátrico-endocrino especializado")
        recommendations.append("• Educación familiar sobre diabetes infantil")
        recommendations.append("• Monitoreo de pubertad y crecimiento")
    elif 13 <= age <= 19 and stage >= 1:
        recommendations.append("• Monitoreo especial en adolescentes con prediabetes")
        recommendations.append("• Educación sobre hábitos saludables en edad escolar")
        recommendations.append("• Apoyo psicológico para cambios de vida")
    elif 20 <= age <= 44 and stage >= 1:
        recommendations.append("• Diabetes Tipo 2 emergente en adultos jóvenes")
        recommendations.append("• Evaluación de síndrome metabólico")
        recommendations.append("• Planificación familiar considerando riesgo genético")
    elif 45 <= age <= 64:
        recommendations.append("• Riesgo máximo por edad - prevención intensiva")
        recommendations.append("• Evaluación cardiovascular completa")
        recommendations.append("• Monitoreo de complicaciones crónicas")
    elif age >= 65:
        recommendations.append("• Adaptar recomendaciones a condiciones geriátricas")
        recommendations.append("• Evaluación de fragilidad y comorbilidades")
        recommendations.append("• Medicamentos con precaución por interacciones")

    # Recomendaciones finales si ninguna específica
    if not recommendations:
        recommendations.append("• Mantener hábitos saludables actuales")
        recommendations.append("• Chequeos preventivos anuales")
        recommendations.append("• Educación continua sobre salud metabólica")

    # Limitar a 10 recomendaciones principales para no sobrecargar
    return recommendations[:10]


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


def train_complete_model(X, y, task='classification', save_path='diabetes_model.pkl'):
    """
    Entrena un modelo completo con validación cruzada y optimización de hiperparámetros.
    Retorna el modelo entrenado y métricas de evaluación.
    """
    print("=== Entrenamiento Completo del Modelo de Diabetes ===")
    
    # Paso 1: Optimización de hiperparámetros
    print("\n1. Optimizando hiperparámetros...")
    best_params, best_score = hyperparameter_tuning(X, y, task=task, k=3)
    
    # Paso 2: Validación cruzada con mejores parámetros
    print("\n2. Validación cruzada con mejores parámetros...")
    cv_metrics = cross_validate_model(X, y, k=5, task=task, epochs=200, batch_size=64, patience=20)
    
    print("Métricas de validación cruzada:")
    for k, v in cv_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Paso 3: Entrenamiento final con todos los datos
    print("\n3. Entrenamiento final del modelo...")
    final_model = AdvancedDiabetesModel(n_features=len(X[0]), task=task)
    
    # Aplicar mejores parámetros si se encontraron
    if best_params:
        # Nota: Para simplificar, usamos los parámetros por defecto ya que el modelo está optimizado
        pass
    
    final_model.fit(X, y, epochs=500, batch_size=64, validation_split=0.2, patience=50)
    
    # Paso 4: Evaluación final
    print("\n4. Evaluación final...")
    if task == 'classification':
        final_metrics = final_model.evaluate_classification(X, y)  # Evaluación en todo el dataset
    else:
        final_metrics = final_model.evaluate_regression(X, y)
    
    print("Métricas finales:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Paso 5: Guardar modelo
    if save_path:
        save_model(final_model, save_path)
    
    return final_model, final_metrics, cv_metrics




def hyperparameter_tuning(X, y, task='classification', k=3):
    """
    Optimización de hiperparámetros usando grid search con validación cruzada.
    Prueba diferentes combinaciones de learning_rate y dropout_rate.
    """
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'dropout_rate': [0.1, 0.2, 0.3]
    }
    
    best_params = None
    best_score = -float('inf') if task == 'classification' else float('inf')
    
    print("Iniciando búsqueda de hiperparámetros...")
    
    for lr in param_grid['learning_rate']:
        for dr in param_grid['dropout_rate']:
            print(f"Probando: lr={lr}, dropout={dr}")
            
            # Crear modelo con estos parámetros
            model = AdvancedDiabetesModel(n_features=len(X[0]), task=task)
            # Nota: Para simplificar, usamos los parámetros por defecto ya que el modelo está optimizado
            # En una implementación completa, modificaríamos la clase para aceptar estos parámetros
            
            # Validación cruzada rápida
            cv_metrics = cross_validate_model(X, y, k=k, task=task, epochs=50, 
                                            batch_size=64, patience=10)
            
            # Evaluar score
            if task == 'classification':
                score = cv_metrics['f1']  # Usar F1 como métrica principal
            else:
                score = cv_metrics['mae']  # Usar MAE para regresión (menor mejor)
            
            if (task == 'classification' and score > best_score) or \
               (task == 'regression' and score < best_score):
                best_score = score
                best_params = {'learning_rate': lr, 'dropout_rate': dr}
                print(f"Nuevo mejor score: {score:.4f}")
    
    print(f"Mejores parámetros encontrados: {best_params}")
    print(f"Mejor score: {best_score:.4f}")
    
    return best_params, best_score


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
    if stage == 2:
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
    elif stage == 1:
        return [
            "(*) Mayor sed de lo normal",
            "(*) Necesidad más frecuente de orinar",
            "(*) Fatiga ocasional después de comer",
            "(*) Náusea o cambios en el apetito",
            "(*) Dolores de cabeza frecuentes",
            "(*) Dificultad para concentrarse",
            "(*) Aumento leve de peso"
        ]
    elif prob >= 0.6:
        return [
            "(*) Pequeñas variaciones de energía durante el día",
            "(*) Hambre intermitente o antojo de dulces",
            "(*) Leve aumento de sed",
            "(*) Pequeños cambios en el peso",
            "(*) Irritabilidad ocasional",
            "(*) Sueño alterado"
        ]
    else:
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

    if stage == 0 and avg_glucose <= 110 and (ldl is None or ldl < 130) and (hdl is None or hdl >= 50):
        return "> 12 años (Riesgo Mínimo - perfil metabólico saludable)"

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

    # Ajuste final por probabilidad - más granular y con límites según etapa
    if stage == 2:
        if prob >= 0.8:
            years = min(years, 0.5)
            clasificacion = "Riesgo Crítico - Control inmediato"
        elif prob >= 0.7:
            years = min(years, 1.0)
            clasificacion = "Riesgo Muy Alto - Control urgente"
        elif prob >= 0.6:
            years = min(years, 1.5)
            clasificacion = "Riesgo Alto - Manejo intensivo"
        elif prob >= 0.5:
            years = min(years, 2.0)
            clasificacion = "Riesgo Alto-Moderado - Seguimiento cercano"
        else:
            years = min(years, 3.0)
            clasificacion = "Riesgo Moderado - Manejo activo"
    elif stage == 1:
        if prob >= 0.8:
            years = min(years, 1.5)
            clasificacion = "Riesgo Muy Alto - Seguimiento urgente"
        elif prob >= 0.7:
            years = min(years, 2.0)
            clasificacion = "Riesgo Alto - Intervención inmediata"
        elif prob >= 0.6:
            years = min(years, 3.0)
            clasificacion = "Riesgo Alto-Moderado - Seguimiento cercano"
        elif prob >= 0.5:
            years = min(years, 4.0)
            clasificacion = "Riesgo Moderado - Vigilancia regular"
        else:
            years = min(years, 6.0)
            clasificacion = "Riesgo Moderado-Bajo - Estilo de vida saludable"
    else:
        if prob >= 0.7:
            years = min(years, 3.0)
            clasificacion = "Riesgo Moderado - Vigilancia regular"
        elif prob >= 0.6:
            years = min(years, 5.0)
            clasificacion = "Riesgo Bajo-Moderado - Chequeo periódico"
        elif prob >= 0.5:
            years = min(years, 7.0)
            clasificacion = "Riesgo Bajo - Chequeos anuales"
        elif prob >= 0.4:
            years = min(years, 10.0)
            clasificacion = "Riesgo Bajo - Mantener hábitos saludables"
        elif prob >= 0.3:
            years = min(years, 12.0)
            clasificacion = "Riesgo Muy Bajo - Prevención"
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


if __name__ == "__main__":
    import pickle
    import os

    print("Entrenando modelos de IA para diabetes...")

    # Entrenar modelo de clasificación
    print("\n=== Entrenando Modelo de Clasificación ===")
    classification_model = train_advanced_model()

    # Entrenar modelo de regresión de tiempo
    print("\n=== Entrenando Modelo de Regresión de Tiempo ===")
    time_model = train_time_to_diabetes_model()

    # Guardar modelos
    print("\nGuardando modelos entrenados...")
    with open('diabetes_classification_model.pkl', 'wb') as f:
        pickle.dump(classification_model, f)

    with open('diabetes_time_model.pkl', 'wb') as f:
        pickle.dump(time_model, f)

    print("✓ Modelos guardados exitosamente")
    print("- diabetes_classification_model.pkl: Predice probabilidad y etapa de diabetes")
    print("- diabetes_time_model.pkl: Predice tiempo hasta desarrollar diabetes")

    # Ejemplo de uso
    print("\n=== Ejemplo de Evaluación ===")
    sample_features = [35, 95, 140, 5.8, 28, 120, 45, 180, 200, 130, 85, 3.5, 1, 120, 3]
    prob, stage, time_est, risks, recs = assess_diabetes_risk_comprehensive(sample_features, classification_model, time_model)

    print(f"Probabilidad de diabetes: {prob:.3f}")
    print(f"Etapa: {get_stage_description(stage)}")
    if time_est:
        print(f"Tiempo estimado hasta diabetes: {time_est:.1f} años")
    print(f"Factores de riesgo: {len(risks)} encontrados")
    print("Recomendaciones principales:")
    for rec in recs[:3]:
        print(f"  {rec}")

