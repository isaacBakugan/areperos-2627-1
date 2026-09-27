# Nombre del integrante: Anthony Caldera
# Cédula del integrante: 30473999

#nota: correr el programa en la cd de tarea-1-codigo, ya que los csv están en la carpeta assets, asi no se corre mas arriba de una carpeta raiz

# ÚNICA LIBRERÍA PERMITIDA POR EL ENUNCIADO Y EL TEST
import matplotlib.pyplot as plt



def funcionSigmoide(z):
    EULER = 2.718281828459045
    return 1.0 / (1.0 + (EULER ** (-z)))

def funcionReLU(z):
    return max(0.0, z)

def cargar_csv(ruta_archivo):
    entradas = []
    salidas_esperadas = []

    with open(ruta_archivo, mode='r', encoding='utf-8') as archivo:
        lineas = archivo.readlines()
        
        for linea in lineas:
            linea_limpia = linea.strip().replace('\ufeff', '')
            if not linea_limpia:
                continue
            
            partes = linea_limpia.split(',')
            try:
                datos_numericos = [float(valor) for valor in partes]
            except ValueError:
                continue  # Ignora encabezados
            
            x = datos_numericos[:-1]
            y = datos_numericos[-1]
            
            entradas.append(x)
            salidas_esperadas.append(y)
            
    return entradas, salidas_esperadas


def calcular_suma(x_vector, pesos, sesgo):
    suma = 0.0
    for i in range(len(x_vector)):
        suma += x_vector[i] * pesos[i]
    suma += sesgo
    return suma

def obtener_predicciones(X, pesos, sesgo, opcion_activacion):
    predicciones = []
    
    for x_vector in X:
        z = calcular_suma(x_vector, pesos, sesgo)
        
        if opcion_activacion == 1:
            y_pred = funcionSigmoide(z)
        else:
            y_pred = funcionReLU(z)
            
        predicciones.append(y_pred)
        
    return predicciones


# -------------------------------------------------------------
# 4. GRAFICACIÓN CON MATPLOTLIB #aca si se uso la IA para poder mostrar el grafico correspondiente
# -------------------------------------------------------------

def graficar_tres_vistas(X, Y_esperado, predicciones, opcion_activacion, nombre_dataset):
    # Tomamos las dos primeras dimensiones para graficar (requisito para n >= 2)
    x1_vals = [fila[0] for fila in X]
    x2_vals = [fila[1] for fila in X]

    # Convertimos predicciones continuas a binarias (0 o 1) para evaluar coincidencia
    y_clase_predicha = []
    for pred in predicciones:
        if opcion_activacion == 1:
            # Sigmoide: Umbral en 0.5
            y_clase_predicha.append(1.0 if pred >= 0.5 else 0.0)
        else:
            # ReLU: Umbral en 0.5 (o >0 según escala)
            y_clase_predicha.append(1.0 if pred >= 0.5 else 0.0)

    # Crear figura con 3 subgráficos (1 fila, 3 columnas)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- GRÁFICO 1: Valor Esperado ---
    esp_0_x1 = [x1_vals[i] for i in range(len(Y_esperado)) if Y_esperado[i] == 0]
    esp_0_x2 = [x2_vals[i] for i in range(len(Y_esperado)) if Y_esperado[i] == 0]
    esp_1_x1 = [x1_vals[i] for i in range(len(Y_esperado)) if Y_esperado[i] == 1]
    esp_1_x2 = [x2_vals[i] for i in range(len(Y_esperado)) if Y_esperado[i] == 1]

    axes[0].scatter(esp_0_x1, esp_0_x2, color='blue', label='Clase 0', alpha=0.7)
    axes[0].scatter(esp_1_x1, esp_1_x2, color='orange', label='Clase 1', alpha=0.7)
    axes[0].set_title("1. Valor Esperado")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].legend()
    axes[0].grid(True)

    # --- GRÁFICO 2: Valor Predicho por el Perceptrón ---
    pred_0_x1 = [x1_vals[i] for i in range(len(y_clase_predicha)) if y_clase_predicha[i] == 0]
    pred_0_x2 = [x2_vals[i] for i in range(len(y_clase_predicha)) if y_clase_predicha[i] == 0]
    pred_1_x1 = [x1_vals[i] for i in range(len(y_clase_predicha)) if y_clase_predicha[i] == 1]
    pred_1_x2 = [x2_vals[i] for i in range(len(y_clase_predicha)) if y_clase_predicha[i] == 1]

    axes[1].scatter(pred_0_x1, pred_0_x2, color='blue', label='Predicho 0', alpha=0.7)
    axes[1].scatter(pred_1_x1, pred_1_x2, color='orange', label='Predicho 1', alpha=0.7)
    axes[1].set_title("2. Valor Predicho")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    axes[1].legend()
    axes[1].grid(True)

    # --- GRÁFICO 3: Coincidencia (Verde = Coinciden, Rojo = No coinciden) ---
    coinciden_x1 = []
    coinciden_x2 = []
    no_coinciden_x1 = []
    no_coinciden_x2 = []

    for i in range(len(Y_esperado)):
        if Y_esperado[i] == y_clase_predicha[i]:
            coinciden_x1.append(x1_vals[i])
            coinciden_x2.append(x2_vals[i])
        else:
            no_coinciden_x1.append(x1_vals[i])
            no_coinciden_x2.append(x2_vals[i])

    axes[2].scatter(coinciden_x1, coinciden_x2, color='green', label='Coinciden', alpha=0.7)
    axes[2].scatter(no_coinciden_x1, no_coinciden_x2, color='red', label='No coinciden', alpha=0.7)
    axes[2].set_title("3. Coincidencia (Verde/Rojo)")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(f"Resultados Perceptrón - {nombre_dataset}", fontsize=14)
    plt.tight_layout()
    plt.show()

def elegir_archivo_csv():
    print("""Seleccione entre los dos archivos csv:
    1. Archivo csv fuzzy_separables
    2. Archivo csv no_separables
    """)
    return int(input("Ingrese 1 o 2: "))

def elegirFuncionActivacion():
    print("""Seleccione la función de activación:
    1. Función Sigmoide
    2. Función ReLU
    """)
    return int(input("Ingrese 1 o 2: "))

def input_w1():
    return float(input("Ingrese el peso w1: "))

def input_w2():
    return float(input("Ingrese el peso w2: "))

def input_b():
    return float(input("Ingrese el valor del sesgo b: "))


# -------------------------------------------------------------
# 6. FLUJO PRINCIPAL #tambien se uso la IA para mostrar resultados aproximados a la consola
# -------------------------------------------------------------

def main():
    while True:
        elegir_csv = elegir_archivo_csv()
        if elegir_csv == 1:
            nombre_ds = "fuzzy_separables"
            X, Y = cargar_csv("assets/fuzzy_separables.csv")
        elif elegir_csv == 2:
            nombre_ds = "no_separables"
            X, Y = cargar_csv("assets/no_separables.csv")
        else:
            print("Opción inválida de CSV.\n")
            continue

        w1 = input_w1()
        w2 = input_w2()
        pesos = [w1, w2]
        b = input_b()

        eleccion_func = elegirFuncionActivacion()
        if eleccion_func not in [1, 2]:
            print("Función inválida.\n")
            continue

        # Obtener las predicciones
        predicciones = obtener_predicciones(X, pesos, b, eleccion_func)

        # Mostrar muestra de resultados por consola
        print("\n" + "="*45)
        print("    RESULTADOS OBTENIDOS DEL PERCEPTRÓN")
        print("="*45)
        print(f"Dataset seleccionado: {nombre_ds}")
        print(f"Pesos aplicados: w1={w1}, w2={w2} | Sesgo b={b}")
        print("-" * 45)
        print("Fila | Entrada (x1, x2) | Esperado (y) | Predicción")
        print("-" * 45)
        
        # Mostrar los primeros 5 resultados
        for i in range(min(5, len(X))):
            print(f" {i+1:<3} | {X[i]} |     {Y[i]}      | {predicciones[i]:.4f}")
        
        print("...")
        print(f"Total de datos procesados: {len(predicciones)}\n")

        # Generar la gráfica requerida
        print("Mostrando gráfica de resultados...")
        graficar_tres_vistas(X, Y, predicciones, eleccion_func, nombre_ds)

        reintentar = input("¿Desea probar con otros datos? (s/n): ").strip().lower()
        if reintentar != 's':
            print("¡Hasta luego!")
            break

if __name__ == "__main__":
    main()