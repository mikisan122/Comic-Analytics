
# Comic Analytics
## Segmentación de clientes mediante Machine Learning

### Descripción del proyecto

Comic Analytics es un proyecto de análisis de datos que utiliza técnicas de Machine Learning para segmentar a los clientes de una tienda de cómics según su comportamiento de compra.

El objetivo es identificar grupos de clientes con características similares para comprender sus hábitos de consumo y facilitar la toma de decisiones comerciales.

### Objetivos

- Explorar y analizar los datos de compra.
- Limpiar y preparar la información.
- Aplicar el algoritmo K-Means.
- Evaluar diferentes cantidades de clústeres.
- Identificar segmentos de clientes.
- Visualizar los resultados mediante un dashboard interactivo.

### Tecnologías utilizadas

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

### Metodología

1. Exploración del dataset.
2. Limpieza y preparación de los datos.
3. Agrupación de compras por cliente.
4. Estandarización de variables con StandardScaler.
5. Aplicación del algoritmo K-Means.
6. Evaluación de clústeres mediante el método del codo y Silhouette Score.
7. Visualización de resultados en un dashboard.

### Resultados

Se analizaron 2,269 registros de compra correspondientes a 600 clientes.

El modelo seleccionado utiliza 2 clústeres:

- **Clúster 0: Coleccionistas frecuentes.** Clientes con mayor número de compras, unidades adquiridas y gasto total.
- **Clúster 1: Lectores ocasionales.** Clientes con menor frecuencia de compra y gasto promedio.

La segmentación permite identificar diferentes comportamientos de consumo dentro de la tienda.

### Estructura del proyecto

```text
A.M proyect/
│
├── codigo/
│   ├── exploracion.py
│   ├── limpieza.py
│   ├── clustering.py
│   └── dashboard.py
│
├── datos/
│   ├── dataset_compras_comics.csv
│   ├── clientes_limpios.csv
│   ├── clientes_escalados.csv
│   └── clientes_segmentados.csv
│
├── documentacion/
│   └── evaluacionJuliaAM.pdf
│
├── resultados/
│   ├── centroides.csv
│   ├── comparacion_modelos.csv
│   ├── estadisticas_por_cluster.csv
│   ├── evaluacion_k.csv
│   └── graficas/
│
├── scaler_comic.pkl
├── requirements.txt
└── README.md
```

### Instalación

Clonar el repositorio:

```bash
git clone URL_DE_TU_REPOSITORIO
```

Entrar a la carpeta:

```bash
cd "A.M proyect"
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

### Ejecución del proyecto

Ejecutar los scripts en el siguiente orden:

```bash
python codigo/exploracion.py
python codigo/limpieza.py
python codigo/clustering.py
```

Para iniciar el dashboard:

```bash
streamlit run codigo/dashboard.py
```

### Autor

Proyecto académico desarrollado como práctica de análisis de datos y Machine Learning.
