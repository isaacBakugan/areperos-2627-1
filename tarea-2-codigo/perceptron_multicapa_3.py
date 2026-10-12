# Nombre del integrante: Samantha Rojas
# Cédula del integrante: 30580398



import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (necesario para la proyección 3D)

TASA_APRENDIZAJE = 0.5
TAMANO_LOTE = 8


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))



# Red neuronal

class PerceptronMulticapa:
    def __init__(self, n_entrada, n_salida, n_ocultas, n_neuronas):
        self.n_entrada = n_entrada
        self.n_salida = n_salida
        self.n_ocultas = n_ocultas      # capas ocultas
        self.n_neuronas = n_neuronas    # neuronas por capa oculta
        self.epocas = 0
        tamanos = [n_entrada] + [n_neuronas] * n_ocultas + [n_salida]
        rng = np.random.default_rng()
        self.W, self.b = [], []
        for fi, fo in zip(tamanos[:-1], tamanos[1:]):
            lim = np.sqrt(6.0 / (fi + fo))
            self.W.append(rng.uniform(-lim, lim, size=(fo, fi)))
            self.b.append(np.zeros(fo))

    # --- alimentación hacia delante ---
    def adelante(self, X):
        A = [X]
        for W, b in zip(self.W, self.b):
            A.append(sigmoide(A[-1] @ W.T + b))
        return A

    def predecir(self, X):
        Y = self.adelante(X)[-1]
        if self.n_salida == 1:
            return (Y[:, 0] > 0.5).astype(int)
        return np.argmax(Y, axis=1)

    # --- retropropagación ---
    def entrenar_epoca(self, X, T):
        n = len(X)
        orden = np.random.permutation(n)
        for i in range(0, n, TAMANO_LOTE):
            idx = orden[i:i + TAMANO_LOTE]
            xb, tb = X[idx], T[idx]
            m = len(idx)
            A = self.adelante(xb)
            delta = (A[-1] - tb) * A[-1] * (1 - A[-1])
            for l in range(len(self.W) - 1, -1, -1):
                gW = delta.T @ A[l] / m
                gb = delta.mean(axis=0)
                if l > 0:
                    delta_prev = (delta @ self.W[l]) * A[l] * (1 - A[l])
                self.W[l] -= TASA_APRENDIZAJE * gW
                self.b[l] -= TASA_APRENDIZAJE * gb
                if l > 0:
                    delta = delta_prev
        Y = self.adelante(X)[-1]
        self.epocas += 1
        return float(np.mean((Y - T) ** 2))

    # --- guardar / cargar ---
    def guardar(self, ruta):
        c = max(self.n_entrada, self.n_neuronas)
        ancho = 3 + c + 1  # 3 columnas + c pesos + columna de sesgo (bias)
        with open(ruta, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Col1", "Col2", "Col3"] + [f"w{i}" for i in range(c)] + ["bias"])
            for letra, valor in [("u", self.n_entrada), ("v", self.n_salida),
                                 ("L", self.n_ocultas), ("b", self.n_neuronas),
                                 ("e", self.epocas)]:
                w.writerow([letra, valor] + [""] * (ancho - 2))
            for l, (W, b) in enumerate(zip(self.W, self.b), start=1):
                tipo = "o" if l == len(self.W) else "h"
                for pos in range(W.shape[0]):
                    pesos = list(W[pos]) + [""] * (c - W.shape[1])
                    w.writerow([tipo, l, pos] + pesos + [b[pos]])

    @staticmethod
    def cargar(ruta):
        hiper, neuronas = {}, {}
        with open(ruta, newline="") as f:
            for fila in csv.reader(f):
                if not fila:
                    continue
                k = fila[0].strip()
                if k in ("u", "v", "L", "b", "e"):
                    hiper[k] = int(float(fila[1]))
                elif k in ("h", "o"):
                    neuronas[(int(fila[1]), int(fila[2]))] = [x.strip() for x in fila[3:]]
        if not neuronas:
            raise ValueError("El archivo no contiene neuronas.")
        capas = sorted({l for l, _ in neuronas})
        tam = {l: max(p for ll, p in neuronas if ll == l) + 1 for l in capas}
        n_entrada = hiper.get("u")
        if n_entrada is None:
            raise ValueError("Falta la fila de hiperparámetros 'u'.")
        sizes = [n_entrada] + [tam[l] for l in capas]
        red = PerceptronMulticapa.__new__(PerceptronMulticapa)
        red.n_entrada = n_entrada
        red.n_salida = sizes[-1]
        red.n_ocultas = len(capas) - 1
        red.n_neuronas = sizes[1] if red.n_ocultas > 0 else 0
        red.epocas = hiper.get("e", 0)
        red.W, red.b = [], []
        for i, l in enumerate(capas):
            fi, fo = sizes[i], sizes[i + 1]
            W, b = np.zeros((fo, fi)), np.zeros(fo)
            for pos in range(fo):
                celdas = neuronas[(l, pos)]
                pesos = celdas[:fi]
                if len(pesos) < fi or any(x == "" for x in pesos):
                    raise ValueError(f"La neurona ({l},{pos}) tiene menos pesos de los esperados.")
                W[pos] = [float(x) for x in pesos]
                resto = [x for x in celdas[fi:] if x != ""]
                if resto:
                    b[pos] = float(resto[-1])  # sesgo = última celda con valor
            red.W.append(W)
            red.b.append(b)
        return red



# Datos

def leer_datos(ruta):
    filas = []
    with open(ruta, newline="") as f:
        for fila in csv.reader(f):
            if not fila or all(x.strip() == "" for x in fila):
                continue
            try:
                filas.append([float(x) for x in fila])
            except ValueError:
                continue  # encabezado
    if not filas:
        raise ValueError("El archivo de datos está vacío.")
    datos = np.array(filas)
    return datos[:, :-1], datos[:, -1]


def preparar(red, ruta):
    X, y = leer_datos(ruta)
    clases = np.unique(y)
    k = len(clases)
    esperado_salida = "1 o 2" if k == 2 else str(k)
    if X.shape[1] != red.n_entrada or not (k == red.n_salida or (k == 2 and red.n_salida == 1)):
        raise ValueError(
            f"La red no corresponde con el archivo. El archivo tiene {X.shape[1]} entradas y "
            f"{k} clases: la red debe configurarse con {X.shape[1]} neuronas de entrada y "
            f"{esperado_salida} neuronas de salida (la red actual tiene "
            f"{red.n_entrada} de entrada y {red.n_salida} de salida)."
        )
    idx = np.searchsorted(clases, y)
    if red.n_salida == 1:
        T = idx[:, None].astype(float)
    else:
        T = np.eye(red.n_salida)[idx]
    return X, idx, T, clases



# Gráficos

def _scatter(X, colores, titulo, cmap=None):
    if X.shape[1] >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colores, cmap=cmap)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    else:
        fig, ax = plt.subplots()
        y2 = X[:, 1] if X.shape[1] == 2 else np.zeros(len(X))
        ax.scatter(X[:, 0], y2, c=colores, cmap=cmap)
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title(titulo)


def graficar(red, X, idx, clases, errores=None):
    pred = red.predecir(X)
    # 1) datos
    _scatter(X, idx, "Datos del archivo (color = clase)", cmap="viridis")
    # 2) correcto / incorrecto
    correcto = pred == idx
    colores = np.where(correcto, "green", "red")
    _scatter(X, colores, f"Predicción: verde = correcta, rojo = incorrecta "
                         f"({correcto.sum()}/{len(idx)})")
    # 3) error o matriz de confusión
    if errores is not None:
        plt.figure()
        plt.plot(range(red.epocas - len(errores) + 1, red.epocas + 1), errores)
        plt.xlabel("Época"); plt.ylabel("Error (MSE)"); plt.title("Error por época")
    else:
        k = len(clases)
        M = np.zeros((k, k), dtype=int)
        for r, p in zip(idx, pred):
            M[r, p] += 1
        fig = plt.figure()
        plt.matshow(M, fignum=fig.number)
        for i in range(k):
            for j in range(k):
                plt.text(j, i, str(M[i, j]), ha="center", va="center", color="red")
        plt.xticks(range(k), [f"{c:g}" for c in clases])
        plt.yticks(range(k), [f"{c:g}" for c in clases])
        plt.xlabel("Predicha"); plt.ylabel("Real")
        plt.title("Matriz de confusión", pad=30)
    plt.show()


def imprimir_hiperparametros(red):
    print("\n=== HIPERPARÁMETROS DE LA RED ===")
    print(f"  Neuronas de entrada (u):      {red.n_entrada}")
    print(f"  Neuronas de salida (v):       {red.n_salida}")
    print(f"  Número de capas ocultas (L):  {red.n_ocultas}")
    print(f"  Neuronas por capa oculta (b): {red.n_neuronas}")
    print(f"  Épocas entrenadas (e):        {red.epocas}")
    print("=================================")



# Interfaz de consola

CARPETA = os.path.dirname(os.path.abspath(__file__))


def linea(titulo=""):
    print("\n" + "=" * 60)
    if titulo:
        print(titulo)
        print("=" * 60)


def leer_entero(mensaje, minimo=1):
    while True:
        try:
            v = int(input(mensaje).strip())
            if v >= minimo:
                return v
        except ValueError:
            pass
        print(f"  -> Escribe un número entero mayor o igual a {minimo}.")


def es_red(ruta):
    """True si el CSV parece un archivo de perceptrón multicapa."""
    try:
        with open(ruta, newline="") as f:
            for i, fila in enumerate(csv.reader(f)):
                if i > 8:
                    break
                if fila and fila[0].strip() in ("u", "v", "L", "b", "e", "h", "o"):
                    return True
    except Exception:
        pass
    return False


def buscar_csv():
    """Busca CSV en la carpeta del programa y en assets/. Devuelve (redes, datos)."""
    redes, datos = [], []
    for carpeta in (CARPETA, os.path.join(CARPETA, "assets")):
        if not os.path.isdir(carpeta):
            continue
        for n in sorted(os.listdir(carpeta)):
            if n.lower().endswith(".csv"):
                ruta = os.path.join(carpeta, n)
                (redes if es_red(ruta) else datos).append(ruta)
    return redes, datos


def elegir_archivo(titulo, candidatos):
    print(f"\n{titulo}")
    if candidatos:
        for i, r in enumerate(candidatos, 1):
            print(f"   {i}. {os.path.relpath(r, CARPETA)}")
    else:
        print("   (no encontré archivos .csv automáticamente)")
    while True:
        s = input("Escribe el NÚMERO del archivo (o su ruta si es otro): ").strip().strip('"')
        if s.isdigit() and 1 <= int(s) <= len(candidatos):
            return candidatos[int(s) - 1]
        if s:
            for ruta in (s, os.path.join(CARPETA, s)):
                if os.path.isfile(ruta):
                    return ruta
        print("  -> No encontré ese archivo, intenta de nuevo.")


def describir(red):
    print(f"\nRed actual: {red.n_entrada} entradas -> {red.n_ocultas} capa(s) oculta(s) de "
          f"{red.n_neuronas} neuronas -> {red.n_salida} salida(s). "
          f"Épocas entrenadas: {red.epocas}.")


def menu_inicial():
    linea("PERCEPTRÓN MULTICAPA")
    while True:
        print("\n¿Qué quieres hacer?")
        print("   1. Crear una red nueva")
        print("   2. Cargar una red guardada")
        op = input("Elige 1 o 2: ").strip()
        if op == "1":
            print("\nConfiguración de la red nueva:")
            n_in = leer_entero("  Neuronas de entrada (= columnas de datos sin la clase; ej. 2 o 3): ")
            n_out = leer_entero("  Neuronas de salida (= número de clases; con 2 clases sirve 1 o 2): ")
            n_oc = leer_entero("  Número de capas ocultas (ej. 1): ")
            n_ne = leer_entero("  Neuronas por capa oculta (ej. 8): ")
            red = PerceptronMulticapa(n_in, n_out, n_oc, n_ne)
            print("\nRed creada.")
            return red
        if op == "2":
            redes, _ = buscar_csv()
            ruta = elegir_archivo("Redes encontradas:", redes)
            try:
                red = PerceptronMulticapa.cargar(ruta)
                print(f"\nRed cargada de {os.path.basename(ruta)}.")
                return red
            except Exception as e:
                print("  -> No pude cargar ese archivo:", e)
        else:
            print("  -> Opción inválida: escribe 1 o 2.")


def main():
    red = menu_inicial()
    while True:
        linea("MENÚ PRINCIPAL")
        describir(red)
        print("\n   1. Entrenar la red (retropropagación)")
        print("   2. Probar la red con datos de prueba")
        print("   3. Guardar la red en un archivo")
        print("   4. Salir")
        op = input("Elige una opción (1-4): ").strip()
        try:
            if op == "1":
                _, datos = buscar_csv()
                ruta = elegir_archivo("Archivos de datos disponibles (elige uno de ENTRENAMIENTO, 'train'):", datos)
                X, idx, T, clases = preparar(red, ruta)
                epocas = leer_entero("¿Cuántas épocas quieres entrenar? (ej. 2000): ")
                print("\nEntrenando...")
                errores = []
                paso = max(1, epocas // 10)
                for e in range(epocas):
                    errores.append(red.entrenar_epoca(X, T))
                    if (e + 1) % paso == 0 or e + 1 == epocas:
                        print(f"   Época {red.epocas:>6}   error = {errores[-1]:.6f}")
                acc = np.mean(red.predecir(X) == idx)
                print(f"\nEntrenamiento terminado. Aciertos sobre los datos de entrenamiento: "
                      f"{acc * 100:.1f}%")
                imprimir_hiperparametros(red)
                print("\nSe abrirán 3 gráficos (datos, correctas/incorrectas, error por época).")
                print("Ciérralos para volver al menú.")
                graficar(red, X, idx, clases, errores=errores)
            elif op == "2":
                _, datos = buscar_csv()
                ruta = elegir_archivo("Archivos de datos disponibles (elige uno de PRUEBA, 'test'):", datos)
                X, idx, T, clases = preparar(red, ruta)
                pred = red.predecir(X)
                acc = np.mean(pred == idx)
                print(f"\nResultado de la prueba: {int((pred == idx).sum())} de {len(idx)} "
                      f"correctas ({acc * 100:.1f}%)")
                imprimir_hiperparametros(red)
                print("\nSe abrirán 3 gráficos (datos, correctas/incorrectas, matriz de confusión).")
                print("Ciérralos para volver al menú.")
                graficar(red, X, idx, clases)
            elif op == "3":
                nombre = input("Nombre del archivo donde guardar (Enter = red_guardada.csv): ").strip().strip('"')
                nombre = nombre or "red_guardada.csv"
                if not nombre.lower().endswith(".csv"):
                    nombre += ".csv"
                ruta = nombre if os.path.isabs(nombre) else os.path.join(CARPETA, nombre)
                red.guardar(ruta)
                print(f"\nRed guardada en: {ruta}")
            elif op == "4":
                print("\n¡Hasta luego!")
                break
            else:
                print("  -> Opción inválida: escribe un número del 1 al 4.")
        except Exception as e:
            print("\n  -> ERROR:", e)


if __name__ == "__main__":
    main()
