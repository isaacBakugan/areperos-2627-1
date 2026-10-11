# ==========================================
# Nombre: Alexandra Padrón 
# Cédula: 30715017
# Tarea 2 — Perceptrón multicapa
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os

# Funciones de activación y derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class MultilayerPerceptron:
    def __init__(self, u, v, L, b, e=0):
        self.u = u  # Entradas
        self.v = v  # Salidas
        self.L = L  # Número de capas ocultas
        self.b = b  # Neuronas por capa oculta
        self.epochs = e
        
        # Inicialización de pesos y sesgos (biases)
        self.weights = []
        self.biases = []
        
        layer_sizes = [self.u] + [self.b] * self.L + [self.v]
        
        for i in range(len(layer_sizes) - 1):
            # Inicialización de He/Xavier básica
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b_vec = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b_vec)

    def forward(self, X):
        self.a_values = [X]
        self.z_values = []
        
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = sigmoid(z)
            self.a_values.append(a)
            
        return self.a_values[-1]

    def backward(self, y_true, lr=0.1):
        m = y_true.shape[0]
        y_pred = self.a_values[-1]
        
        # Error en la capa de salida (MSE derivado)
        dz = (y_pred - y_true) * sigmoid_derivative(self.z_values[-1])
        
        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(self.a_values[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * sigmoid_derivative(self.z_values[i-1])
                
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    def train(self, X, y, epochs, lr=0.1):
        errors = []
        for _ in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean(np.square(y_pred - y))
            errors.append(loss)
            self.backward(y, lr)
            self.epochs += 1
        return errors

    def predict(self, X):
        return self.forward(X)

    def save(self, filename):
        max_cols = max(self.u, self.b)
        header = ["Col1", "Col2", "Col3"] + [f"w{i}" for i in range(max_cols)] + ["bias"]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            # Hiperparámetros
            writer.writerow(["u", self.u] + [""] * (len(header)-2))
            writer.writerow(["v", self.v] + [""] * (len(header)-2))
            writer.writerow(["L", self.L] + [""] * (len(header)-2))
            writer.writerow(["b", self.b] + [""] * (len(header)-2))
            writer.writerow(["e", self.epochs] + [""] * (len(header)-2))
            
            # Capas ocultas
            for l in range(self.L):
                for n in range(self.b):
                    row = ["h", l+1, n]
                    w = self.weights[l][:, n].tolist()
                    b_val = [self.biases[l][0, n]]
                    # Rellenar con vacíos si hay menos pesos que el max_cols
                    padding = [""] * (max_cols - len(w))
                    writer.writerow(row + w + padding + b_val)
                    
            # Capa de salida
            for n in range(self.v):
                row = ["o", self.L+1, n]
                w = self.weights[-1][:, n].tolist()
                b_val = [self.biases[-1][0, n]]
                padding = [""] * (max_cols - len(w))
                writer.writerow(row + w + padding + b_val)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Saltar encabezado
            
            u = int(next(reader)[1])
            v = int(next(reader)[1])
            L = int(next(reader)[1])
            b = int(next(reader)[1])
            e = int(next(reader)[1])
            
            mlp = cls(u, v, L, b, e)
            
            for row in reader:
                if not row: continue
                n_type = row[0]
                l_idx = int(row[1]) - 1
                n_idx = int(row[2])
                
                if n_type == 'h':
                    weights = [float(x) for x in row[3:3+mlp.weights[l_idx].shape[0]]]
                    bias = float(row[-1])
                    mlp.weights[l_idx][:, n_idx] = weights
                    mlp.biases[l_idx][0, n_idx] = bias
                elif n_type == 'o':
                    weights = [float(x) for x in row[3:3+mlp.weights[-1].shape[0]]]
                    bias = float(row[-1])
                    mlp.weights[-1][:, n_idx] = weights
                    mlp.biases[-1][0, n_idx] = bias
                    
        return mlp

# --- Funciones auxiliares ---

def encode_labels(y, unique_classes):
    mapping = {val: i for i, val in enumerate(unique_classes)}
    y_encoded = np.array([mapping[val] for val in y])
    y_onehot = np.zeros((y.size, len(unique_classes)))
    y_onehot[np.arange(y.size), y_encoded] = 1
    return y_onehot, mapping

def load_data(filename, expected_u):
    df = pd.read_csv(filename)
    if df.shape[1] - 1 != expected_u:
        raise ValueError(f"El archivo tiene {df.shape[1]-1} entradas, pero la red espera {expected_u}.")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def plot_results(X, y_true, y_pred, errors=None, conf_matrix=None, is_train=True):
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Gráfico de dispersión (2D o 3D)
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask
    
    if X.shape[1] >= 3:
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(X[correct_mask, 0], X[correct_mask, 1], X[correct_mask, 2], c='g', label='Correcto', marker='o')
        ax1.scatter(X[incorrect_mask, 0], X[incorrect_mask, 1], X[incorrect_mask, 2], c='r', label='Incorrecto', marker='x')
        ax1.set_title("Datos (3D)")
    else:
        ax1 = fig.add_subplot(131)
        ax1.scatter(X[correct_mask, 0], X[correct_mask, 1], c='g', label='Correcto', marker='o')
        ax1.scatter(X[incorrect_mask, 0], X[incorrect_mask, 1], c='r', label='Incorrecto', marker='x')
        ax1.set_title("Datos (2D)")
    ax1.legend()

    # 2. Correctos vs Incorrectos
    ax2 = fig.add_subplot(132)
    labels = ['Correctos', 'Incorrectos']
    counts = [np.sum(correct_mask), np.sum(incorrect_mask)]
    ax2.bar(labels, counts, color=['green', 'red'])
    ax2.set_title("Resultados de Predicción")

    # 3. Error / Matriz de confusión
    ax3 = fig.add_subplot(133)
    if is_train and errors is not None:
        ax3.plot(errors, color='blue')
        ax3.set_title("Error vs Épocas")
        ax3.set_xlabel("Época")
        ax3.set_ylabel("MSE")
    elif not is_train and conf_matrix is not None:
        cax = ax3.matshow(conf_matrix, cmap='Blues')
        fig.colorbar(cax)
        ax3.set_title("Matriz de Confusión")
        
    plt.tight_layout()
    plt.show()

def get_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# --- Menú Principal ---

def main():
    mlp = None
    unique_classes = None
    
    print("=== Perceptrón Multicapa ===")
    print("1. Crear nuevo perceptrón")
    print("2. Cargar de archivo")
    opt = input("Seleccione una opción: ")
    
    if opt == '1':
        u = int(input("Número de neuronas de entrada: "))
        v = int(input("Número de neuronas de salida (clases): "))
        L = int(input("Número de capas ocultas: "))
        b = int(input("Número de neuronas por capa oculta: "))
        mlp = MultilayerPerceptron(u, v, L, b)
    elif opt == '2':
        filename = input("Archivo del modelo (CSV): ")
        mlp = MultilayerPerceptron.load(filename)
        print("Modelo cargado exitosamente.")
    else:
        print("Opción no válida. Saliendo.")
        return

    while True:
        print("\n--- MENÚ ---")
        print("1. Entrenar red")
        print("2. Probar red")
        print("3. Guardar red")
        print("4. Salir")
        
        # Imprimir Hiperparámetros actuales
        print(f"\n[Hiperparámetros] Entradas:{mlp.u} | Salidas:{mlp.v} | Capas ocultas:{mlp.L} | Neuronas/capa:{mlp.b} | Épocas entrenadas:{mlp.epochs}")
        
        choice = input("Opción: ")
        
        if choice == '1':
            file_train = input("Archivo de datos de entrenamiento (CSV): ")
            try:
                X, y = load_data(file_train, mlp.u)
                if unique_classes is None:
                    unique_classes = np.unique(y)
                
                y_onehot, _ = encode_labels(y, unique_classes)
                ep = int(input("Número de épocas: "))
                
                print("Entrenando...")
                errors = mlp.train(X, y_onehot, ep, lr=0.05)
                
                # Predicciones para gráficas
                preds = mlp.predict(X)
                y_pred_classes = np.argmax(preds, axis=1)
                y_true_classes = np.argmax(y_onehot, axis=1)
                
                plot_results(X, y_true_classes, y_pred_classes, errors=errors, is_train=True)
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '2':
            file_test = input("Archivo de datos de prueba (CSV): ")
            try:
                X, y = load_data(file_test, mlp.u)
                if unique_classes is None:
                    unique_classes = np.unique(y)
                    
                y_onehot, _ = encode_labels(y, unique_classes)
                
                preds = mlp.predict(X)
                y_pred_classes = np.argmax(preds, axis=1)
                y_true_classes = np.argmax(y_onehot, axis=1)
                
                cm = get_confusion_matrix(y_true_classes, y_pred_classes, mlp.v)
                plot_results(X, y_true_classes, y_pred_classes, conf_matrix=cm, is_train=False)
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '3':
            filename = input("Nombre de archivo para guardar (ej. modelo.csv): ")
            mlp.save(filename)
            print("Red guardada exitosamente.")
            
        elif choice == '4':
            print("Saliendo...")
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    main()