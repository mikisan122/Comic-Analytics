
# Comic Analytics - Parte 2: Exploracion del dataset
# Ejecutar desde la carpeta principal con:
# python codigo/exploracion.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =====================================================
# 1. CONFIGURACION DE RUTAS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "datos" / "dataset_compras_comics.csv"
SALIDA = BASE_DIR / "resultados" / "graficas"

SALIDA.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. CARGAR DATOS
# =====================================================

if not DATASET.exists():
    raise FileNotFoundError(
        f"No se encontro el dataset: {DATASET}"
    )

df = pd.read_csv(DATASET, parse_dates=["Fecha_Compra"])

print("\n========== PRIMEROS 10 REGISTROS ==========")
print(df.head(10).to_string(index=False))

print("\n========== DIMENSIONES ==========")
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")

print("\n========== NOMBRES DE COLUMNAS ==========")
print(df.columns.tolist())

print("\n========== TIPOS DE DATOS ==========")
print(df.dtypes)

print("\n========== ESTADISTICAS DESCRIPTIVAS ==========")
print(df.describe(include="all").to_string())

print("\n========== VALORES NULOS ==========")
print(df.isnull().sum())

print("\n========== REGISTROS DUPLICADOS ==========")
print(df.duplicated().sum())

print("\n========== RESUMEN GENERAL ==========")
print(f"Clientes diferentes: {df['Cliente_ID'].nunique()}")
print(f"Editoriales: {df['Editorial'].nunique()}")
print(f"Generos de comic: {df['Genero_Comic'].nunique()}")
print(f"Ventas totales: ${df['Total_Compra'].sum():,.2f}")

# =====================================================
# 3. CONFIGURAR ESTILO DE GRAFICAS
# =====================================================

sns.set_theme(style="whitegrid")

# =====================================================
# 4. GRAFICA 1: COMPRAS POR EDITORIAL
# =====================================================

plt.figure(figsize=(9, 5))

orden_editoriales = df["Editorial"].value_counts().index

sns.countplot(
    data=df,
    x="Editorial",
    order=orden_editoriales
)

plt.title("Cantidad de compras por editorial")
plt.xlabel("Editorial")
plt.ylabel("Numero de compras")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    SALIDA / "01_compras_por_editorial.png",
    dpi=150
)

plt.show()
plt.close()

# =====================================================
# 5. GRAFICA 2: DISTRIBUCION DEL GASTO
# =====================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Total_Compra",
    bins=25,
    kde=True
)

plt.title("Distribucion del total de compra")
plt.xlabel("Total de compra ($)")
plt.ylabel("Frecuencia")
plt.tight_layout()

plt.savefig(
    SALIDA / "02_distribucion_gasto.png",
    dpi=150
)

plt.show()
plt.close()

# =====================================================
# 6. GRAFICA 3: COMPRAS POR GENERO
# =====================================================

plt.figure(figsize=(9, 5))

orden_generos = df["Genero_Comic"].value_counts().index

sns.countplot(
    data=df,
    y="Genero_Comic",
    order=orden_generos
)

plt.title("Cantidad de compras por genero de comic")
plt.xlabel("Numero de compras")
plt.ylabel("Genero")
plt.tight_layout()

plt.savefig(
    SALIDA / "03_compras_por_genero.png",
    dpi=150
)

plt.show()
plt.close()

# =====================================================
# 7. FINALIZAR
# =====================================================

print("\n========== PROCESO COMPLETADO ==========")
print(f"Las tres graficas se guardaron en: {SALIDA}")
