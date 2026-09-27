# Nombre del integrante: Alexandra Padrón
# Cédula del integrante: 30715017

import matplotlib.pyplot as plt


def funcion_activacion_escalon(z):
    """
    Función de activación Escalón unitario (Heaviside).
    Devuelve 1 si z >= 0, de lo contrario 0.
    """
    return 1 if z >= 0 else 0


def funcion_activacion_signo(z):
    """
    Función de activación Signo (Bipolar).
    Devuelve 1 si z >= 0, de lo contrario -1.
    """
    return 1 if z >= 0 else -1


def calcular_suma_ponderada(x_vector, pesos, sesgo):
    """
    Calcula la suma ponderada z = sesgo + sum(w_i * x_i) desde cero.
    """
    z = sesgo
    for i in range(len(x_vector)):
        z += pesos[i] * x_vector[i]
    return z


def cargar_csv(ruta_archivo):
    
    entradas = []
    salidas_esperadas = []

    with open(ruta_archivo, mode="r", encoding="utf-8") as archivo:
        lineas = archivo.readlines()
        
        # Omitir la primera línea (el encabezado) y procesar el resto
        for linea in lineas[1:]:
            linea = linea.strip()
            if not linea:
                continue
            
            # Separar el texto por comas manualmente
            fila = linea.split(",")
            valores = [float(val) for val in fila]
            
            entradas.append(valores[:-1])
            salidas_esperadas.append(valores[-1])

    return entradas, salidas_esperadas



def graficar_resultados(
    entradas, salidas_esperadas, predicciones, coincidencias
):
    """
    Muestra tres gráficos de dispersión (scatter):
    1. Valor esperado
    2. Valor predicho por el perceptrón
    3. Coincidencias (Verde si coinciden, Rojo si no)
    
    Si n > 3, grafica solo las dos primeras dimensiones de entrada.
    """
    # Tomar las primeras dos dimensiones del vector de entrada (x1, x2)
    x1 = [vec[0] for vec in entradas]
    x2 = (
        [vec[1] for vec in entradas]
        if len(entradas[0]) > 1
        else [0] * len(entradas)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gráfico 1: Valor esperado
    scatter1 = axes[0].scatter(
        x1, x2, c=salidas_esperadas, cmap="coolwarm", edgecolors="k"
    )
    axes[0].set_title("Valor Esperado")
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")
    fig.colorbar(scatter1, ax=axes[0])

    # Gráfico 2: Valor predicho
    scatter2 = axes[1].scatter(
        x1, x2, c=predicciones, cmap="coolwarm", edgecolors="k"
    )
    axes[1].set_title("Valor Predicho")
    axes[1].set_xlabel("X1")
    axes[1].set_ylabel("X2")
    fig.colorbar(scatter2, ax=axes[1])

    # Gráfico 3: Coincidencias (Verde = Coinciden, Rojo = No coinciden)
    colores_coincidencia = [
        "green" if coincide else "red" for coincide in coincidencias
    ]
    axes[2].scatter(x1, x2, c=colores_coincidencia, edgecolors="k")
    axes[2].set_title("Coincidencia (Verde=Acierto, Rojo=Fallo)")
    axes[2].set_xlabel("X1")
    axes[2].set_ylabel("X2")

    plt.tight_layout()
    plt.show()


def main():
    print("=========================================")
    print("   SIMULADOR DE PERCEPTRÓN INTERACTIVO   ")
    print("=========================================\n")

    ruta_csv = input(
        "Ingrese la ruta del archivo CSV (ej. assets/no_separables.csv): "
    ).strip()

    try:
        entradas, salidas_esperadas = cargar_csv(ruta_csv)
    except Exception as e:
        print(f"Error al abrir o procesar el archivo CSV: {e}")
        return

    n_dimensiones = len(entradas[0])
    print(
        f"\nCSV cargado correctamente. Cada vector tiene {n_dimensiones} dimensiones de entrada."
    )

    mantenimiento = True
    while mantenimiento:
        print("\n--- Configuración de Pesos ---")
        sesgo = float(input("Ingrese el peso para el sesgo (bias w0): "))

        pesos = []
        for i in range(n_dimensiones):
            peso = float(input(f"Ingrese el peso w{i+1} para la dimensión {i+1}: "))
            pesos.append(peso)

        print("\n--- Seleccione la Función de Activación ---")
        print("1. Función Escalón (Heaviside: 1 si z >= 0, 0 si z < 0)")
        print("2. Función Signo (Bipolar: 1 si z >= 0, -1 si z < 0)")
        opcion_activacion = input("Selección (1 o 2): ").strip()

        predicciones = []
        coincidencias = []

        # Proceso de ejecución vector a vector
        for i in range(len(entradas)):
            z = calcular_suma_ponderada(entradas[i], pesos, sesgo)

            if opcion_activacion == "2":
                pred = funcion_activacion_signo(z)
            else:
                pred = funcion_activacion_escalon(z)

            predicciones.append(pred)
            coincidencias.append(pred == salidas_esperadas[i])

        aciertos = sum(coincidencias)
        total = len(coincidencias)
        porcentaje = (aciertos / total) * 100
        print(f"\nExactitud obtenida: {aciertos}/{total} ({porcentaje:.2f}%)")

        # Visualización de los 3 gráficos mediante matplotlib
        graficar_resultados(
            entradas, salidas_esperadas, predicciones, coincidencias
        )

        # Permitir al usuario reintentar con otros pesos
        respuesta = (
            input("\n¿Desea probar con un conjunto de pesos distinto? (s/n): ")
            .strip()
            .lower()
        )
        if respuesta != "s":
            mantenimiento = False
            print("\n¡Ejecución finalizada!")


if __name__ == "__main__":
    main()