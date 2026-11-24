
# Comic Analytics - Parte 4: Implementacion de K-Means

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================================
# 1. CONFIGURACION DE RUTAS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CARPETA_DATOS = BASE_DIR / "datos"
CARPETA_RESULTADOS = BASE_DIR / "resultados"
CARPETA_GRAFICAS = CARPETA_RESULTADOS / "graficas"

CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)

ARCHIVO_CLIENTES = CARPETA_DATOS / "clientes_limpios.csv"
ARCHIVO_ESCALADOS = CARPETA_DATOS / "clientes_escalados.csv"
ARCHIVO_SCALER = BASE_DIR / "scaler_comic.pkl"

# Variables utilizadas durante la preparacion de datos
VARIABLES = [
    "Numero_Compras",
    "Unidades_Totales",
    "Gasto_Total",
    "Precio_Promedio"
]

# =====================================================
# 2. VERIFICAR ARCHIVOS
# =====================================================

for archivo in [ARCHIVO_CLIENTES, ARCHIVO_ESCALADOS, ARCHIVO_SCALER]:
    if not archivo.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo necesario: {archivo}"
        )

# =====================================================
# 3. CARGAR LOS DATOS
# =====================================================

clientes = pd.read_csv(ARCHIVO_CLIENTES)
escalados = pd.read_csv(ARCHIVO_ESCALADOS)
scaler = joblib.load(ARCHIVO_SCALER)

if not all(columna in escalados.columns for columna in VARIABLES):
    raise ValueError("Faltan variables necesarias en clientes_escalados.csv.")

if len(clientes) != len(escalados):
    raise ValueError("Los archivos limpios y escalados no tienen la misma cantidad de registros.")

X = escalados[VARIABLES]

if len(X) < 9:
    raise ValueError("Se necesitan al menos 9 clientes para evaluar K de 2 a 8.")

if X.isnull().any().any():
    raise ValueError("Hay valores nulos en los datos escalados. Revisa limpieza.py.")

print("========== DATOS PARA K-MEANS ==========")
print(f"Clientes: {len(clientes)}")
print(f"Variables: {', '.join(VARIABLES)}")

# =====================================================
# 4. EVALUAR DIFERENTES VALORES DE K
# =====================================================

resultados = []
modelos = {}

for k in range(2, 9):

    modelo = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        random_state=42
    )

    etiquetas = modelo.fit_predict(X)
    silueta = silhouette_score(X, etiquetas)

    resultados.append({
        "K": k,
        "Inercia": modelo.inertia_,
        "Coeficiente_Silhouette": silueta
    })

    modelos[k] = modelo

    print(
        f"K={k}: Inercia={modelo.inertia_:.3f} | "
        f"Silhouette={silueta:.4f}"
    )

evaluacion = pd.DataFrame(resultados)

evaluacion.to_csv(
    CARPETA_RESULTADOS / "evaluacion_k.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 5. GRAFICAS DE EVALUACION
# =====================================================

sns.set_theme(style="whitegrid")

# Metodo del codo
plt.figure(figsize=(8, 5))

plt.plot(
    evaluacion["K"],
    evaluacion["Inercia"],
    marker="o"
)

plt.title("Metodo del codo - Comic Analytics")
plt.xlabel("Numero de clusters (K)")
plt.ylabel("Inercia")
plt.xticks(evaluacion["K"])
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "04_metodo_del_codo.png",
    dpi=160
)

plt.close()

# Coeficiente Silhouette
plt.figure(figsize=(8, 5))

plt.plot(
    evaluacion["K"],
    evaluacion["Coeficiente_Silhouette"],
    marker="o"
)

plt.title("Evaluacion Silhouette por numero de clusters")
plt.xlabel("Numero de clusters (K)")
plt.ylabel("Coeficiente Silhouette")
plt.xticks(evaluacion["K"])
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "05_silhouette_por_k.png",
    dpi=160
)

plt.close()

# =====================================================
# 6. SELECCIONAR EL MEJOR MODELO
# =====================================================

k_final = int(
    evaluacion.loc[
        evaluacion["Coeficiente_Silhouette"].idxmax(),
        "K"
    ]
)

modelo_final = modelos[k_final]

clientes["Cluster"] = modelo_final.labels_

print("\n========== MODELO FINAL ==========")
print(f"K seleccionado: {k_final}")
print("Motivo: obtuvo el mayor coeficiente Silhouette.")

# =====================================================
# 7. COMPARAR K=2 Y K=3
# =====================================================

comparacion = evaluacion[
    evaluacion["K"].isin([2, 3])
].copy()

comparacion.to_csv(
    CARPETA_RESULTADOS / "comparacion_modelos.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== COMPARACION K=2 VS K=3 ==========")
print(comparacion.to_string(index=False))

# =====================================================
# 8. CLIENTES POR CLUSTER
# =====================================================

conteo = clientes["Cluster"].value_counts().sort_index()

print("\n========== CLIENTES POR CLUSTER ==========")

for cluster, cantidad in conteo.items():

    porcentaje = cantidad / len(clientes) * 100

    print(
        f"Cluster {cluster}: {cantidad} clientes "
        f"({porcentaje:.1f}%)"
    )

# =====================================================
# 9. CENTROIDES EN ESCALA ORIGINAL
# =====================================================

centroides_originales = scaler.inverse_transform(
    modelo_final.cluster_centers_
)

centroides = pd.DataFrame(
    centroides_originales,
    columns=VARIABLES
)

centroides.insert(0, "Cluster", range(k_final))
centroides = centroides.round(2)

centroides.to_csv(
    CARPETA_RESULTADOS / "centroides.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== CENTROIDES ==========")
print(centroides.to_string(index=False))

# =====================================================
# 10. ESTADISTICAS POR CLUSTER
# =====================================================

estadisticas = clientes.groupby("Cluster")[VARIABLES].agg(
    ["count", "mean", "median", "min", "max"]
).round(2)

estadisticas.to_csv(
    CARPETA_RESULTADOS / "estadisticas_por_cluster.csv",
    encoding="utf-8-sig"
)

# =====================================================
# 11. GRAFICA DE SEGMENTACION
# =====================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=clientes,
    x="Numero_Compras",
    y="Gasto_Total",
    hue="Cluster",
    palette="Set2",
    s=65
)

plt.title(f"Segmentacion de clientes con K-Means (K={k_final})")
plt.xlabel("Numero de compras")
plt.ylabel("Gasto total ($)")
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "06_segmentacion_clientes.png",
    dpi=160
)

plt.close()

# =====================================================
# 12. GRAFICA DE CLIENTES POR CLUSTER
# =====================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=clientes,
    x="Cluster",
    hue="Cluster",
    legend=False,
    palette="Set2"
)

plt.title("Cantidad de clientes por cluster")
plt.xlabel("Cluster")
plt.ylabel("Cantidad de clientes")
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "07_clientes_por_cluster.png",
    dpi=160
)

plt.close()

# =====================================================
# 13. GRAFICA DE GASTO POR CLUSTER
# =====================================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=centroides,
    x="Cluster",
    y="Gasto_Total",
    hue="Cluster",
    legend=False,
    palette="Set2"
)

plt.title("Gasto total promedio de los centroides")
plt.xlabel("Cluster")
plt.ylabel("Gasto total promedio ($)")
plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "08_centroides_gasto.png",
    dpi=160
)

plt.close()

# =====================================================
# 14. GUARDAR DATASET FINAL
# =====================================================

clientes.to_csv(
    CARPETA_DATOS / "clientes_segmentados.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 15. FINALIZAR
# =====================================================

print("\n========== ARCHIVOS GENERADOS ==========")

print("resultados/evaluacion_k.csv")
print("resultados/comparacion_modelos.csv")
print("resultados/centroides.csv")
print("resultados/estadisticas_por_cluster.csv")
print("datos/clientes_segmentados.csv")
print("resultados/graficas/")

print("\nProceso K-Means terminado correctamente.")
