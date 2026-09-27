# Nombre del integrante: Samantha Rojas
# Cédula del integrante: 30580398

import matplotlib.pyplot as plt


#  Rutas de los datasets
def obtener_carpeta_script():
    ruta_completa = __file__
    idx = max(ruta_completa.rfind("/"), ruta_completa.rfind("\\"))
    if idx == -1:
        return ""
    return ruta_completa[: idx + 1]


CARPETA_SCRIPT = obtener_carpeta_script()

DATASETS_DISPONIBLES = [
    CARPETA_SCRIPT + "assets/no_separables.csv",
    CARPETA_SCRIPT + "assets/fuzzy_separables.csv",
]


# lectura del csv 
def leer_csv(ruta):
    entradas = []   # lista de listas: cada una con los n-1 valores x
    esperados = []  # lista de floats: el valor y de cada fila

    with open(ruta, "r", encoding="utf-8") as archivo:
        lineas = archivo.readlines()

    if not lineas:
        return entradas, esperados

    # Detectar si la primera línea es encabezado 
    primera = lineas[0].strip().split(",")
    tiene_encabezado = False
    for valor in primera:
        try:
            float(valor)
        except ValueError:
            tiene_encabezado = True
            break

    inicio = 1 if tiene_encabezado else 0

    for linea in lineas[inicio:]:
        linea = linea.strip()
        if linea == "":
            continue
        valores = [float(v) for v in linea.split(",")]
        entradas.append(valores[:-1])
        esperados.append(valores[-1])

    return entradas, esperados


# funcion suma combinacion lineal
def suma_ponderada(entrada, pesos, sesgo):
    total = sesgo
    for xi, wi in zip(entrada, pesos):
        total += xi * wi
    return total


# funciones de activacion
def escalon(z):
    """Función escalón (Heaviside): salida binaria 0 / 1."""
    return 1.0 if z >= 0 else 0.0


def signo(z):
    """Función signo: salida bipolar -1 / 1."""
    return 1.0 if z >= 0 else -1.0


ACTIVACIONES = {
    "1": ("escalon (salidas 0/1)", escalon),
    "2": ("signo (salidas -1/1)", signo),
}


#  Perceptrón
def predecir(entrada, pesos, sesgo, funcion_activacion):
    z = suma_ponderada(entrada, pesos, sesgo)
    return funcion_activacion(z)



#  Lectura de pesos y función de activación 

def pedir_float(mensaje):
    while True:
        texto = input(mensaje).strip()
        try:
            return float(texto)
        except ValueError:
            print("  -> Valor invalido, escribe un numero (ej: 0.5, -1, 2).")


def pedir_pesos(n_entradas):
    print(f"\nEl dataset tiene {n_entradas} columna(s) de entrada.")
    sesgo = pedir_float("Ingresa el peso del sesgo (bias): ")
    pesos = []
    for i in range(n_entradas):
        w = pedir_float(f"Ingresa el peso w{i + 1} (para x{i + 1}): ")
        pesos.append(w)
    return sesgo, pesos


def pedir_activacion():
    print("\nFunciones de activacion disponibles:")
    for clave, (nombre, _) in ACTIVACIONES.items():
        print(f"  {clave}) {nombre}")
    while True:
        opcion = input("Elige una funcion de activacion (1/2): ").strip()
        if opcion in ACTIVACIONES:
            nombre, funcion = ACTIVACIONES[opcion]
            print(f"  -> Usando: {nombre}")
            return funcion
        print("  -> Opcion invalida.")


# graficos mathplotlib

def graficar(entradas, esperados, predichos):
    # Si hay mas de 2 dimensiones, solo se grafican las dos primeras
    x1 = [fila[0] if len(fila) > 0 else 0.0 for fila in entradas]
    x2 = [fila[1] if len(fila) > 1 else 0.0 for fila in entradas]

    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))

    # Grafico 1: valores esperados
    ejes[0].scatter(x1, x2, c=esperados, cmap="coolwarm")
    ejes[0].set_title("Valores esperados")
    ejes[0].set_xlabel("x1")
    ejes[0].set_ylabel("x2")

    # Grafico 2: valores predichos
    ejes[1].scatter(x1, x2, c=predichos, cmap="coolwarm")
    ejes[1].set_title("Valores predichos")
    ejes[1].set_xlabel("x1")
    ejes[1].set_ylabel("x2")

    # Grafico 3: aciertos (verde) / fallos (rojo)
    colores = []
    for esperado, predicho in zip(esperados, predichos):
        colores.append("green" if esperado == predicho else "red")
    ejes[2].scatter(x1, x2, c=colores)
    ejes[2].set_title("Aciertos (verde) / Fallos (rojo)")
    ejes[2].set_xlabel("x1")
    ejes[2].set_ylabel("x2")

    plt.tight_layout()
    plt.show()



#  Selección de dataset
def elegir_dataset():
    print("\nDatasets disponibles:")
    for i, ruta in enumerate(DATASETS_DISPONIBLES, start=1):
        print(f"  {i}) {ruta}")
    opcion_otra = len(DATASETS_DISPONIBLES) + 1
    print(f"  {opcion_otra}) Escribir otra ruta")

    opcion = input(
        f"Elige un dataset [1-{opcion_otra}] (Enter = 1): "
    ).strip()

    if opcion == "":
        return DATASETS_DISPONIBLES[0]

    if opcion.isdigit():
        indice = int(opcion)
        if 1 <= indice <= len(DATASETS_DISPONIBLES):
            return DATASETS_DISPONIBLES[indice - 1]
        if indice == opcion_otra:
            return input("Escribe la ruta del CSV: ").strip()

    print("  -> Opcion invalida, se usara el dataset 1 por defecto.")
    return DATASETS_DISPONIBLES[0]


# main
def main():
    print("=== Tarea 1: Perceptron ===")
    ruta = elegir_dataset()
    print(f"Cargando: {ruta}")

    try:
        entradas, esperados = leer_csv(ruta)
    except FileNotFoundError:
        print(f"No se encontro el archivo: {ruta}")
        return

    if not entradas:
        print("El archivo esta vacio o no se pudo leer.")
        return

    n_entradas = len(entradas[0])

    seguir = True
    while seguir:
        sesgo, pesos = pedir_pesos(n_entradas)
        funcion_activacion = pedir_activacion()

        predichos = [
            predecir(fila, pesos, sesgo, funcion_activacion) for fila in entradas
        ]

        aciertos = sum(1 for e, p in zip(esperados, predichos) if e == p)
        print(f"\nAciertos: {aciertos}/{len(entradas)}")

        graficar(entradas, esperados, predichos)

        respuesta = input("\n¿Deseas probar con otros pesos? (s/n): ").strip().lower()
        seguir = respuesta == "s"

    print("Fin del programa.")


if __name__ == "__main__":
    main()