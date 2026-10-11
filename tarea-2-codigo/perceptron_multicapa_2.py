# Nombre del integrante: Anthony Caldera
# Cédula del integrante: 30473999

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))

def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def derivada_sigmoide(a):
    return a * (1.0 - a)

class PerceptronMulticapa:
    def __init__(self, u, v, L, b, e=0):
        self.u = int(u)  # Entradas
        self.v = int(v)  # Salidas (clases)
        self.L = int(L)  # Número de capas ocultas
        self.b = int(b)  # Neuronas por capa oculta
        self.e = int(e)  # Épocas entrenadas
        
        # Estructura: capa de entrada + capas ocultas + capa de salida
        tamanios = [self.u] + [self.b] * self.L + [self.v]
        
        # Inicialización de pesos y sesgos
        self.pesos = []
        self.sesgos = []
        for i in range(len(tamanios) - 1):
            n_in, n_out = tamanios[i], tamanios[i + 1]
            # Inicialización estándar
            w = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            sesgo = np.zeros((1, n_out))
            self.pesos.append(w)
            self.sesgos.append(sesgo)

    def forward(self, X):
        self.activaciones = [X]
        a = X
        for i in range(len(self.pesos)):
            z = np.dot(a, self.pesos[i]) + self.sesgos[i]
            a = sigmoide(z)
            self.activaciones.append(a)
        return self.activaciones[-1]

    def backward(self, Y_onehot, lr=0.1):
        m = Y_onehot.shape[0]
        y_pred = self.activaciones[-1]
        
        # Error en la capa de salida
        delta = (y_pred - Y_onehot) * derivada_sigmoide(y_pred)
        
        # Retropropagación capa por capa hacia atrás
        for i in range(len(self.pesos) - 1, -1, -1):
            grad_w = np.dot(self.activaciones[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.pesos[i].T) * derivada_sigmoide(self.activaciones[i])
                
            self.pesos[i] -= lr * grad_w
            self.sesgos[i] -= lr * grad_b

    def entrenar(self, X, Y_onehot, epocas, lr=0.1):
        errores = []
        for _ in range(epocas):
            indices = np.random.permutation(len(X))
            for i in indices:
                self.forward(X[i:i+1])
                self.backward(Y_onehot[i:i+1], lr)
            
            pred = self.forward(X)
            loss = np.mean((pred - Y_onehot) ** 2)
            errores.append(loss)
            self.e += 1
        return errores

    def predecir(self, X):
        return self.forward(X)

    def clasificar(self, X):
        preds = self.predecir(X)
        return np.argmax(preds, axis=1)

    def guardar(self, ruta):
        c = max(self.u, self.b)
        encabezado = ["Col1", "Col2", "Col3"] + [f"w{i}" for i in range(c)] + ["bias"]
        
        with open(ruta, mode='w', newline='', encoding='utf-8') as f:
            escritor = csv.writer(f)
            escritor.writerow(encabezado)
            
            # Primeras 5 filas: hiperparámetros
            for letra, val in [("u", self.u), ("v", self.v), ("L", self.L), ("b", self.b), ("e", self.e)]:
                escritor.writerow([letra, val] + [""] * (len(encabezado) - 2))
                
            # Filas de neuronas ocultas ('h')
            for l in range(self.L):
                for n in range(self.b):
                    pesos = self.pesos[l][:, n].tolist()
                    sesgo = [self.sesgos[l][0, n]]
                    relleno = [""] * (c - len(pesos))
                    escritor.writerow(["h", l + 1, n] + pesos + relleno + sesgo)
                    
            # Filas de neuronas de salida ('o')
            for n in range(self.v):
                pesos = self.pesos[-1][:, n].tolist()
                sesgo = [self.sesgos[-1][0, n]]
                relleno = [""] * (c - len(pesos))
                escritor.writerow(["o", self.L + 1, n] + pesos + relleno + sesgo)

    @classmethod
    def cargar(cls, ruta):
        ruta_res = resolver_ruta(ruta)
        with open(ruta_res, mode='r', encoding='utf-8') as f:
            filas = [fila for fila in csv.reader(f) if fila and any(col.strip() for col in fila)]
            
        hiper = {fila[0].strip(): int(float(fila[1])) for fila in filas[1:6]}
        red = cls(hiper["u"], hiper["v"], hiper["L"], hiper["b"], hiper["e"])
        
        for fila in filas[6:]:
            tipo, capa, pos = fila[0].strip(), int(fila[1]) - 1, int(fila[2])
            n_pesos = red.pesos[capa].shape[0]
            red.pesos[capa][:, pos] = [float(w) for w in fila[3:3 + n_pesos]]
            # El sesgo está en la última columna de la fila
            red.sesgos[capa][0, pos] = float(fila[-1])
            
        return red

    def mostrar_hiperparametros(self):
        print("\n" + "=" * 45)
        print("    HIPERPARÁMETROS DEL PERCEPTRÓN")
        print("=" * 45)
        print(f"  u (entradas)          : {self.u}")
        print(f"  v (salidas / clases)  : {self.v}")
        print(f"  L (capas ocultas)     : {self.L}")
        print(f"  b (neuronas por capa) : {self.b}")
        print(f"  e (épocas acumuladas) : {self.e}")
        print("=" * 45)



def resolver_ruta(ruta):
    ruta = ruta.strip().strip('"').strip("'")
    if os.path.exists(ruta):
        return ruta
    alt1 = os.path.join(CARPETA_ACTUAL, ruta)
    if os.path.exists(alt1):
        return alt1
    alt2 = os.path.join(CARPETA_ACTUAL, "assets", ruta)
    if os.path.exists(alt2):
        return alt2
    raise FileNotFoundError(f"No se encontró el archivo '{ruta}'.")

def cargar_datos(ruta, red):
    ruta_res = resolver_ruta(ruta)
    filas = []
    with open(ruta_res, mode='r', encoding='utf-8') as f:
        for linea in f:
            partes = linea.strip().replace('\ufeff', '').split(',')
            try:
                filas.append([float(x) for x in partes])
            except ValueError:
                continue  # Salta encabezados
                
    if not filas:
        raise ValueError("El archivo no contiene filas con datos numéricos.")
        
    matriz = np.array(filas)
    X = matriz[:, :-1]
    y_raw = matriz[:, -1]
    
    # Validación con la red
    u_archivo = X.shape[1]
    clases_unicas = np.unique(y_raw)
    v_archivo = len(clases_unicas)
    
    if red.u != u_archivo or red.v != v_archivo:
        raise ValueError(
            f"El archivo no coincide con la red configurada:\n"
            f"  - La red espera: u={red.u} entradas y v={red.v} salidas.\n"
            f"  - El archivo tiene: {u_archivo} entradas y {v_archivo} clases ({list(clases_unicas)}).\n"
            f"  -> Configure la red con: u={u_archivo} neuronas de entrada y v={v_archivo} neuronas de salida."
        )
        
    # Codificación One-Hot
    mapeo = {c: i for i, c in enumerate(clases_unicas)}
    y_indices = np.array([mapeo[c] for c in y_raw])
    Y_onehot = np.zeros((len(y_raw), red.v))
    Y_onehot[np.arange(len(y_raw)), y_indices] = 1.0
    
    return X, y_raw, y_indices, Y_onehot, clases_unicas


#graficos

def graficar(X, y_raw, y_indices, y_pred, clases_unicas, errores=None, matriz_confusion=None):
    fig = plt.figure(figsize=(15, 5))
    es_3d = X.shape[1] >= 3
    correctos = (y_indices == y_pred)
    
    # --- Gráfico 1: Datos del archivo (2D o 3D) ---
    ax1 = fig.add_subplot(1, 3, 1, projection='3d') if es_3d else fig.add_subplot(1, 3, 1)
    colores = plt.cm.tab10(np.arange(len(clases_unicas)))
    for i, c in enumerate(clases_unicas):
        mask = (y_raw == c)
        if es_3d:
            ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2], color=colores[i], label=f"Clase {int(c)}", alpha=0.7)
        else:
            ax1.scatter(X[mask, 0], X[mask, 1], color=colores[i], label=f"Clase {int(c)}", alpha=0.7)
    ax1.set_title("1. Datos del Archivo")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    if es_3d: ax1.set_zlabel("x3")
    ax1.legend()
    
    # --- Gráfico 2: Predicciones Correctas vs Incorrectas ---
    ax2 = fig.add_subplot(1, 3, 2, projection='3d') if es_3d else fig.add_subplot(1, 3, 2)
    coords_corr = (X[correctos, 0], X[correctos, 1], X[correctos, 2]) if es_3d else (X[correctos, 0], X[correctos, 1])
    coords_inc = (X[~correctos, 0], X[~correctos, 1], X[~correctos, 2]) if es_3d else (X[~correctos, 0], X[~correctos, 1])
    
    ax2.scatter(*coords_corr, color='green', label=f"Correcto ({np.sum(correctos)})", alpha=0.7)
    ax2.scatter(*coords_inc, color='red', marker='x', label=f"Incorrecto ({np.sum(~correctos)})", alpha=0.9)
    ax2.set_title(f"2. Predicción: Correcto / Incorrecto")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    if es_3d: ax2.set_zlabel("x3")
    ax2.legend()
    
    # --- Gráfico 3: Error de épocas (Entrenamiento) o Matriz de Confusión (Prueba) ---
    ax3 = fig.add_subplot(1, 3, 3)
    if errores is not None:
        ax3.plot(errores, color='blue')
        ax3.set_title("3. Error de la Red vs Épocas")
        ax3.set_xlabel("Época")
        ax3.set_ylabel("Error Cuadrático Medio (MSE)")
        ax3.grid(True)
    elif matriz_confusion is not None:
        cax = ax3.matshow(matriz_confusion, cmap='Blues')
        fig.colorbar(cax, ax=ax3, fraction=0.046, pad=0.04)
        etiquetas = [str(int(c)) for c in clases_unicas]
        ax3.set_xticks(range(len(clases_unicas)))
        ax3.set_yticks(range(len(clases_unicas)))
        ax3.set_xticklabels(etiquetas)
        ax3.set_yticklabels(etiquetas)
        ax3.set_xlabel("Clase Predicha")
        ax3.set_ylabel("Clase Real")
        ax3.set_title("3. Matriz de Confusión")
        for i in range(len(clases_unicas)):
            for j in range(len(clases_unicas)):
                ax3.text(j, i, str(matriz_confusion[i, j]), ha='center', va='center',
                         color='white' if matriz_confusion[i, j] > matriz_confusion.max() / 2 else 'black')
                         
    plt.tight_layout()
    plt.show()


#menu main

def menu_inicial():
    while True:
        print("\n=============================================")
        print("       PERCEPTRÓN MULTICAPA - TAREA 2")
        print("=============================================")
        print("  1. Crear un nuevo perceptrón multicapa")
        print("  2. Cargar un perceptrón desde archivo")
        opc = input("Ingrese 1 o 2: ").strip()
        
        if opc == '1':
            print("\n--- Configurar nuevo perceptrón ---")
            u = int(input("Número de neuronas de entrada: "))
            v = int(input("Número de neuronas de salida: "))
            L = int(input("Número de capas ocultas: "))
            b = int(input("Número de neuronas por capa: "))
            return PerceptronMulticapa(u, v, L, b)
        elif opc == '2':
            ruta = input("Archivo del perceptrón (CSV, ej. red_2d_2.csv): ").strip()
            try:
                red = PerceptronMulticapa.cargar(ruta)
                print("Perceptrón cargado exitosamente.")
                return red
            except Exception as e:
                print(f"Error al cargar: {e}")
        else:
            print("Opción inválida.")

def main():
    red = menu_inicial()
    red.mostrar_hiperparametros()
    
    while True:
        print("\n--------------- MENÚ ---------------")
        print("  1. Entrenar la red")
        print("  2. Probar la red")
        print("  3. Guardar la red")
        print("  4. Salir")
        opc = input("Seleccione una opción: ").strip()
        
        try:
            if opc == '1':
                ruta = input("Archivo de entrenamiento (CSV, ej. 2-d_2-class_train.csv): ")
                X, y_raw, y_indices, Y_onehot, clases = cargar_datos(ruta, red)
                epocas = int(input("Número de épocas: "))
                
                print(f"Entrenando {epocas} épocas...")
                errores = red.entrenar(X, Y_onehot, epocas, lr=0.1)
                y_pred = red.clasificar(X)
                
                print(f"Aciertos: {np.sum(y_indices == y_pred)} / {len(X)}")
                red.mostrar_hiperparametros()
                graficar(X, y_raw, y_indices, y_pred, clases, errores=errores)
                
            elif opc == '2':
                ruta = input("Archivo de prueba (CSV, ej. 2-d_2-class_test.csv): ")
                X, y_raw, y_indices, Y_onehot, clases = cargar_datos(ruta, red)
                
                y_pred = red.clasificar(X)
                print(f"Aciertos: {np.sum(y_indices == y_pred)} / {len(X)}")
                
                # Matriz de confusión
                cm = np.zeros((len(clases), len(clases)), dtype=int)
                for real, pred in zip(y_indices, y_pred):
                    cm[real, pred] += 1
                    
                red.mostrar_hiperparametros()
                graficar(X, y_raw, y_indices, y_pred, clases, matriz_confusion=cm)
                
            elif opc == '3':
                ruta = input("Nombre para guardar (CSV, ej. red_guardada.csv): ").strip()
                red.guardar(ruta)
                print(f"Red guardada exitosamente en '{ruta}'.")
                
            elif opc == '4':
                print("¡Hasta luego!")
                break
            else:
                print("Opción inválida.")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
