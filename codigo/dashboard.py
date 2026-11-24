
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# 1. CONFIGURACION
# =====================================================

st.set_page_config(
    page_title="Comic Analytics",
    page_icon="📚",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

ARCHIVO_VENTAS = BASE_DIR / "datos" / "dataset_compras_comics.csv"
ARCHIVO_CLIENTES = BASE_DIR / "datos" / "clientes_segmentados.csv"

# =====================================================
# 2. ESTILOS
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #101114;
    color: #F3F3F3;
}

[data-testid="stSidebar"] {
    background-color: #191A20;
}

.main-title {
    color: #E63946;
    font-size: 2.5rem;
    font-weight: 800;
}

.subtitle {
    color: #C7C7C7;
    margin-top: -12px;
}

div[data-testid="stMetric"] {
    background: #202127;
    border: 1px solid #383943;
    padding: 15px;
    border-radius: 12px;
}

h2, h3 {
    color: #F3F3F3;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. CARGAR DATASETS
# =====================================================

@st.cache_data
def cargar_datos():

    ventas = pd.read_csv(ARCHIVO_VENTAS)
    clientes = pd.read_csv(ARCHIVO_CLIENTES)

    ventas["Cliente_ID"] = ventas["Cliente_ID"].astype(str)
    clientes["Cliente_ID"] = clientes["Cliente_ID"].astype(str)

    # Comprobar la columna del cluster
    posibles_clusters = [
        "Cluster", "cluster", "CLUSTER",
        "Clusters", "clusters"
    ]

    columna_cluster = next(
        (col for col in posibles_clusters if col in clientes.columns),
        None
    )

    if columna_cluster is None:
        raise ValueError(
            "No se encontro la columna Cluster "
            "en clientes_segmentados.csv"
        )

    if columna_cluster != "Cluster":
        clientes.rename(
            columns={columna_cluster: "Cluster"},
            inplace=True
        )

    # Columnas necesarias
    columnas_requeridas = [
        "Cliente_ID",
        "Edad",
        "Genero_Favorito",
        "Numero_Compras",
        "Unidades_Totales",
        "Gasto_Total",
        "Precio_Promedio",
        "Cluster"
    ]

    faltantes = [
        col for col in columnas_requeridas
        if col not in clientes.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas: {faltantes}"
        )

    # Convertir datos numericos
    columnas_numericas = [
        "Edad",
        "Numero_Compras",
        "Unidades_Totales",
        "Gasto_Total",
        "Precio_Promedio",
        "Cluster"
    ]

    for columna in columnas_numericas:
        clientes[columna] = pd.to_numeric(
            clientes[columna],
            errors="coerce"
        )

    ventas["Total_Compra"] = pd.to_numeric(
        ventas["Total_Compra"],
        errors="coerce"
    )

    if clientes[columnas_numericas].isnull().any().any():
        raise ValueError(
            "Existen valores invalidos en las variables numericas "
            "del archivo de clientes."
        )

    clientes["Cluster"] = clientes["Cluster"].astype(int)

    # Relacionar las ventas con su cluster
    ventas = ventas.merge(
        clientes[["Cliente_ID", "Cluster"]],
        on="Cliente_ID",
        how="inner"
    )

    return ventas, clientes


# =====================================================
# 4. TITULO DEL DASHBOARD
# =====================================================

st.markdown(
    '<div class="main-title">COMIC ANALYTICS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Segmentacion de clientes de una tienda de comics '
    'mediante K-Means'
    '</div>',
    unsafe_allow_html=True
)

# =====================================================
# 5. VERIFICAR ARCHIVOS Y CARGAR DATOS
# =====================================================

if not ARCHIVO_VENTAS.exists():
    st.error("No se encontro dataset_compras_comics.csv")
    st.stop()

if not ARCHIVO_CLIENTES.exists():
    st.error("No se encontro clientes_segmentados.csv")
    st.stop()

try:
    ventas, clientes = cargar_datos()

except Exception as error:
    st.error(f"Error al cargar los datos: {error}")
    st.stop()


# =====================================================
# 6. FILTROS INTERACTIVOS
# =====================================================

st.sidebar.title("FILTROS")

# Filtro de edad
edad_minima = int(clientes["Edad"].min())
edad_maxima = int(clientes["Edad"].max())

rango_edad = st.sidebar.slider(
    "Rango de edad",
    min_value=edad_minima,
    max_value=edad_maxima,
    value=(edad_minima, edad_maxima)
)

# Filtro de genero favorito
generos = sorted(
    clientes["Genero_Favorito"]
    .dropna()
    .astype(str)
    .unique()
)

generos_disponibles = ["Todos"] + generos

genero_seleccionado = st.sidebar.selectbox(
    "Genero favorito",
    generos_disponibles
)

# Filtro de cluster
clusters_disponibles = sorted(
    clientes["Cluster"].unique()
)

clusters_seleccionados = st.sidebar.multiselect(
    "Segmento de clientes",
    options=clusters_disponibles,
    default=clusters_disponibles,
    format_func=lambda valor: f"Cluster {valor}"
)

# Aplicar filtro de edad
clientes_filtrados = clientes[
    clientes["Edad"].between(
        rango_edad[0],
        rango_edad[1]
    )
].copy()

# Aplicar filtro de genero
if genero_seleccionado != "Todos":

    clientes_filtrados = clientes_filtrados[
        clientes_filtrados["Genero_Favorito"]
        == genero_seleccionado
    ]

# Aplicar filtro de cluster
clientes_filtrados = clientes_filtrados[
    clientes_filtrados["Cluster"].isin(
        clusters_seleccionados
    )
]

# CORRECCION IMPORTANTE:
# Esta variable se define fuera de las dos paginas
# para que ambas puedan utilizarla.

total_clientes = clientes_filtrados[
    "Cliente_ID"
].nunique()

# Verificar que existan resultados
if clientes_filtrados.empty:

    st.warning(
        "No hay clientes que coincidan con los filtros seleccionados."
    )

    st.stop()

# Filtrar ventas de los clientes seleccionados
ventas_filtradas = ventas[
    ventas["Cliente_ID"].isin(
        clientes_filtrados["Cliente_ID"]
    )
].copy()


# =====================================================
# 7. MENU DE NAVEGACION
# =====================================================

st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "NAVEGACION",
    [
        "1. Analisis general",
        "2. Segmentacion K-Means"
    ]
)


# =====================================================
# PAGINA 1: ANALISIS GENERAL
# =====================================================

if pagina == "1. Analisis general":

    st.header("Analisis general")

    # ---------------------------------------------
    # INDICADORES KPI
    # ---------------------------------------------

    total_compras = len(ventas_filtradas)

    ingresos = ventas_filtradas[
        "Total_Compra"
    ].sum()

    promedio_compra = ventas_filtradas[
        "Total_Compra"
    ].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total de clientes",
        f"{total_clientes:,}"
    )

    c2.metric(
        "Total de compras",
        f"{total_compras:,}"
    )

    c3.metric(
        "Ingresos",
        f"${ingresos:,.2f}"
    )

    c4.metric(
        "Promedio por compra",
        f"${promedio_compra:,.2f}"
    )

    # ---------------------------------------------
    # DISTRIBUCION POR CLUSTER
    # ---------------------------------------------

    st.subheader("Distribucion de clientes por cluster")

    conteo = (
        clientes_filtrados["Cluster"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    etiquetas = [
        f"Cluster {int(valor)}"
        for valor in conteo.index
    ]

    ax.bar(
        etiquetas,
        conteo.values,
        color="#E63946"
    )

    ax.set_ylabel("Cantidad de clientes")
    ax.set_xlabel("Segmento")
    ax.set_title("Clientes por cluster")
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    # ---------------------------------------------
    # COMPRAS POR EDITORIAL
    # ---------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Compras por editorial")

        editorial = ventas_filtradas[
            "Editorial"
        ].value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            editorial.index.astype(str),
            editorial.values,
            color="#E63946"
        )

        ax.set_ylabel("Numero de compras")
        ax.set_xlabel("Editorial")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------
    # COMPRAS POR GENERO
    # ---------------------------------------------

    with col2:

        st.subheader("Compras por genero de comic")

        genero = ventas_filtradas[
            "Genero_Comic"
        ].value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.barh(
            genero.index.astype(str),
            genero.values,
            color="#457B9D"
        )

        ax.set_xlabel("Numero de compras")
        ax.set_ylabel("Genero")
        ax.grid(axis="x", alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------
    # TABLA GENERAL
    # ---------------------------------------------

    st.subheader("Resumen de clientes")

    columnas_tabla = [
        "Cliente_ID",
        "Edad",
        "Genero_Favorito",
        "Numero_Compras",
        "Unidades_Totales",
        "Gasto_Total",
        "Cluster"
    ]

    st.dataframe(
        clientes_filtrados[columnas_tabla].sort_values(
            "Gasto_Total",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# PAGINA 2: SEGMENTACION K-MEANS
# =====================================================

elif pagina == "2. Segmentacion K-Means":

    st.header("Segmentacion mediante K-Means")

    st.caption(
        "Modelo final implementado con K=2 y cuatro "
        "variables de comportamiento de compra."
    )

    # ---------------------------------------------
    # GRAFICA DE DISPERSION
    # ---------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Agrupamientos de clientes")

        fig, ax = plt.subplots(figsize=(7, 5))

        colores = {
            0: "#E63946",
            1: "#457B9D"
        }

        for cluster, grupo in clientes_filtrados.groupby(
            "Cluster"
        ):

            ax.scatter(
                grupo["Numero_Compras"],
                grupo["Gasto_Total"],
                label=f"Cluster {int(cluster)}",
                alpha=0.75,
                s=35,
                color=colores.get(
                    int(cluster),
                    "#2A9D8F"
                )
            )

        ax.set_xlabel("Numero de compras")
        ax.set_ylabel("Gasto total por cliente ($)")
        ax.set_title("Compras vs gasto total")
        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------
    # COMPRAS Y UNIDADES POR CLUSTER
    # ---------------------------------------------

    with col2:

        st.subheader("Compras y unidades por grupo")

        resumen = clientes_filtrados.groupby(
            "Cluster"
        )[
            [
                "Numero_Compras",
                "Unidades_Totales"
            ]
        ].mean()

        fig, ax = plt.subplots(figsize=(7, 5))

        resumen.plot(
            kind="bar",
            ax=ax,
            color=["#E63946", "#457B9D"]
        )

        ax.set_xlabel("Cluster")
        ax.set_ylabel("Promedio por cliente")
        ax.set_title("Actividad de compra por segmento")

        ax.set_xticklabels(
            [
                f"Cluster {int(x)}"
                for x in resumen.index
            ],
            rotation=0
        )

        ax.legend([
            "Compras",
            "Unidades"
        ])

        ax.grid(axis="y", alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------
    # GASTO Y PRECIO PROMEDIO
    # ---------------------------------------------

    st.subheader("Comparacion de gastos por cluster")

    resumen_dinero = clientes_filtrados.groupby(
        "Cluster"
    )[
        [
            "Gasto_Total",
            "Precio_Promedio"
        ]
    ].mean()

    fig, ax = plt.subplots(figsize=(9, 4))

    resumen_dinero.plot(
        kind="bar",
        ax=ax,
        color=["#E63946", "#F4A261"]
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Monto ($)")
    ax.set_title(
        "Gasto total y precio promedio por cliente"
    )

    ax.set_xticklabels(
        [
            f"Cluster {int(x)}"
            for x in resumen_dinero.index
        ],
        rotation=0
    )

    ax.legend([
        "Gasto total promedio",
        "Precio promedio por comic"
    ])

    ax.grid(axis="y", alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    # ---------------------------------------------
    # ESTADISTICAS DESCRIPTIVAS
    # ---------------------------------------------

    st.subheader("Estadisticas descriptivas por cluster")

    estadisticas = clientes_filtrados.groupby(
        "Cluster"
    ).agg(

        Clientes=("Cliente_ID", "nunique"),

        Edad_Promedio=("Edad", "mean"),

        Compras_Promedio=(
            "Numero_Compras",
            "mean"
        ),

        Unidades_Promedio=(
            "Unidades_Totales",
            "mean"
        ),

        Gasto_Promedio=(
            "Gasto_Total",
            "mean"
        ),

        Precio_Promedio=(
            "Precio_Promedio",
            "mean"
        )

    ).round(2)

    st.dataframe(
        estadisticas,
        use_container_width=True
    )

    # ---------------------------------------------
    # DESCRIPCION DE LOS SEGMENTOS
    # ---------------------------------------------

    st.subheader("Descripcion de los segmentos")

    descripciones = {

        0: (
            "Coleccionistas frecuentes",
            "Presentan mas compras, adquieren mas unidades "
            "y registran un gasto promedio elevado. "
            "Se recomienda ofrecer preventas, novedades "
            "y beneficios de fidelidad."
        ),

        1: (
            "Lectores ocasionales",
            "Realizan menos compras y tienen un gasto "
            "promedio menor. Se recomienda crear paquetes, "
            "promociones y recomendaciones de comics."
        )
    }

    for cluster, grupo in clientes_filtrados.groupby(
        "Cluster"
    ):

        nombre, descripcion = descripciones.get(
            int(cluster),
            (
                f"Segmento {int(cluster)}",
                "Grupo identificado mediante K-Means."
            )
        )

        cantidad = grupo["Cliente_ID"].nunique()

        # Utiliza el total de clientes filtrados
        porcentaje = (
            cantidad / total_clientes
        ) * 100

        st.markdown(
            f"""
            **{nombre} (Cluster {int(cluster)})**

            - Clientes: {cantidad}
            - Porcentaje: {porcentaje:.1f}%

            {descripcion}
            """
        )

    # ---------------------------------------------
    # HALLAZGOS
    # ---------------------------------------------

    st.subheader("Principales hallazgos")

    st.markdown("""
    1. Los lectores ocasionales representan el grupo
       mas numeroso de clientes.

    2. Los coleccionistas frecuentes muestran mayor
       actividad de compra y gasto.

    3. La segmentacion permite adaptar las promociones
       y recomendaciones a cada grupo.
    """)

    # ---------------------------------------------
    # RECOMENDACIONES
    # ---------------------------------------------

    st.subheader("Recomendaciones")

    st.markdown("""
    1. Crear un programa de puntos y preventas
       para los coleccionistas frecuentes.

    2. Ofrecer paquetes y descuentos para incentivar
       nuevas compras de lectores ocasionales.

    3. Utilizar las preferencias de genero y editorial
       para personalizar las recomendaciones.
    """)


# =====================================================
# 8. PIE DE PAGINA
# =====================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Comic Analytics | Proyecto integrador de mineria de datos"
)

st.sidebar.caption(
    "Algoritmo de aprendizaje no supervisado: K-Means"
)
