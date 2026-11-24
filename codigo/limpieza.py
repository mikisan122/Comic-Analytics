
# Comic Analytics - Parte 3: Limpieza y preparacion de datos
# Ejecutar desde la carpeta principal con:
# python codigo/limpieza.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

# =====================================================
# 1. CONFIGURACION DE RUTAS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "datos" / "dataset_compras_comics.csv"
CARPETA_DATOS = BASE_DIR / "datos"
ARCHIVO_SCALER = BASE_DIR / "scaler_comic.pkl"

# =====================================================
# 2. CARGAR DATASET
# =====================================================

if not DATASET.exists():
    raise FileNotFoundError(
        f"No se encontro el dataset: {DATASET}"
    )

df = pd.read_csv(DATASET)

print("DATOS ORIGINALES:", df.shape)

# =====================================================
# 3. CORREGIR TIPOS DE DATOS
# =====================================================

columnas = [
    "Edad",
    "Precio_Unitario",
    "Cantidad",
    "Total_Compra"
]

for col in columnas:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df["Fecha_Compra"] = pd.to_datetime(
    df["Fecha_Compra"],
    errors="coerce"
)

# =====================================================
# 4. ELIMINAR DUPLICADOS Y DATOS INCOMPLETOS
# =====================================================

duplicados = df.duplicated().sum()

df = df.drop_duplicates()

df = df.dropna(
    subset=[
        "Cliente_ID",
        "Edad",
        "Precio_Unitario",
        "Cantidad",
        "Fecha_Compra"
    ]
)

# =====================================================
# 5. ELIMINAR VALORES IMPOSIBLES
# =====================================================

df = df[
    (df["Edad"].between(1, 100)) &
    (df["Precio_Unitario"] > 0) &
    (df["Cantidad"] > 0)
].copy()

# =====================================================
# 6. RECALCULAR TOTAL DE COMPRA
# =====================================================

df["Total_Compra"] = (
    df["Precio_Unitario"] * df["Cantidad"]
).round(2)

# =====================================================
# 7. AGRUPAR COMPRAS POR CLIENTE
# =====================================================

clientes = df.groupby("Cliente_ID").agg(
    Edad=("Edad", "first"),
    Genero_Favorito=("Genero_Favorito", "first"),
    Numero_Compras=("Cliente_ID", "size"),
    Unidades_Totales=("Cantidad", "sum"),
    Gasto_Total=("Total_Compra", "sum"),
    Precio_Promedio=("Precio_Unitario", "mean")
).reset_index()

clientes["Gasto_Total"] = clientes["Gasto_Total"].round(2)
clientes["Precio_Promedio"] = clientes["Precio_Promedio"].round(2)

# =====================================================
# 8. DETECTAR VALORES ATIPICOS MEDIANTE IQR
# =====================================================

variables = [
    "Numero_Compras",
    "Unidades_Totales",
    "Gasto_Total",
    "Precio_Promedio"
]

print("\nDETECCION DE VALORES ATIPICOS")

for col in variables:

    q1 = clientes[col].quantile(0.25)
    q3 = clientes[col].quantile(0.75)
    iqr = q3 - q1

    limite_min = q1 - 1.5 * iqr
    limite_max = q3 + 1.5 * iqr

    atipicos = clientes[
        (clientes[col] < limite_min) |
        (clientes[col] > limite_max)
    ]

    print(f"{col}: {len(atipicos)} valores atipicos")

# Conservamos los valores atipicos que sean compras reales,
# porque pueden representar clientes coleccionistas.

# =====================================================
# 9. MOSTRAR RESULTADOS
# =====================================================

print("\nRESULTADOS DE LIMPIEZA")

print("Clientes:", len(clientes))
print("Duplicados eliminados:", duplicados)

print("\nValores nulos:")
print(clientes.isnull().sum())

print("\nPrimeros 10 clientes:")
print(clientes.head(10))

# =====================================================
# 10. ESCALAR VARIABLES CON STANDARDSCALER
# =====================================================

scaler = StandardScaler()

datos_escalados = scaler.fit_transform(
    clientes[variables]
)

clientes_escalados = pd.DataFrame(
    datos_escalados,
    columns=variables
)

clientes_escalados.insert(
    0,
    "Cliente_ID",
    clientes["Cliente_ID"]
)

# =====================================================
# 11. GUARDAR RESULTADOS
# =====================================================

clientes.to_csv(
    CARPETA_DATOS / "clientes_limpios.csv",
    index=False
)

clientes_escalados.to_csv(
    CARPETA_DATOS / "clientes_escalados.csv",
    index=False
)

joblib.dump(
    scaler,
    ARCHIVO_SCALER
)

print("\nARCHIVOS GENERADOS CORRECTAMENTE")
print("datos/clientes_limpios.csv")
print("datos/clientes_escalados.csv")
print("scaler_comic.pkl")

print("\nPROCESO COMPLETADO")
