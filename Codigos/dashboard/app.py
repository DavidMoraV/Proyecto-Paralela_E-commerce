"""
app.py
Módulo 5 (M5) — Dashboard, Visualización e Integración Final
Sistema de Recomendación Paralelo para E-Commerce — RetailRocket Dataset

Corre con: python app.py
Abre en el navegador: http://127.0.0.1:8050
"""
from pathlib import Path

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pandas as pd


# Rutas (ajustar si la estructura de carpetas es distinta)

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_PROCESSED = RAIZ / "datos"
CARPETA_RESULTADOS = RAIZ / "resultados"


# Carga de datos (una sola vez, al iniciar la app)

eventos = pl.read_parquet(CARPETA_PROCESSED / "eventos_limpios.parquet")
segmentos = pl.read_parquet(CARPETA_PROCESSED / "user_segments.parquet")
recomendaciones = pd.read_csv(CARPETA_RESULTADOS / "recomendaciones_muestra.csv")

resumen_speedup = pd.read_csv(CARPETA_RESULTADOS / "resumen_speedup_por_etapa.csv")
speedup_etl = pd.read_csv(CARPETA_RESULTADOS / "speedup_etl.csv")

# La prueba de carga es de Entrega 3 -- se carga de forma tolerante, ya que
# el dashboard debe poder abrir aunque todavia no se haya corrido.
RUTA_PRUEBA_CARGA = CARPETA_RESULTADOS / "entrega3" / "prueba_carga.csv"
prueba_carga_df = pd.read_csv(RUTA_PRUEBA_CARGA) if RUTA_PRUEBA_CARGA.exists() else None

NOMBRES_CLUSTER = {
    0: "Cluster 0 — Bajo/moderado",
    1: "Cluster 1 — Ocasional (1 interacción)",
    2: "Cluster 2 — Muy activo",
    3: "Cluster 3 — Sobre el promedio",
    4: "Cluster 4 — Orientado a compra",
    5: "Cluster 5 — Actividad intermedia",
}


# Precómputo de agregados livianos para los gráficos (evita recalcular en cada callback)

eventos_por_tipo = (
    eventos.group_by("event").agg(pl.len().alias("cantidad")).sort("cantidad", descending=True).to_pandas()
)
top_productos = (
    eventos.group_by("itemid").agg(pl.len().alias("interacciones"))
    .sort("interacciones", descending=True).head(15).to_pandas()
)
top_productos["itemid"] = top_productos["itemid"].astype(str)

actividad_hora = (
    eventos.group_by("hora_del_dia").agg(pl.len().alias("interacciones")).sort("hora_del_dia").to_pandas()
)

cluster_size = (
    segmentos.group_by("cluster").agg(pl.len().alias("usuarios")).sort("cluster").to_pandas()
)
cluster_size["cluster_nombre"] = cluster_size["cluster"].map(NOMBRES_CLUSTER)

perfil_clusters = (
    segmentos.group_by("cluster")
    .agg([
        pl.mean("total_interacciones").alias("Interacciones"),
        pl.mean("productos_distintos").alias("Productos distintos"),
        pl.mean("views").alias("Views"),
        pl.mean("carritos").alias("Carritos"),
        pl.mean("compras").alias("Compras"),
    ])
    .sort("cluster").to_pandas()
)

usuarios_con_recomendaciones = sorted(recomendaciones["visitorid"].unique().tolist())


# App

app = dash.Dash(__name__)
app.title = "Sistema de Recomendación — RetailRocket"

COLORES = {"gris": "#5F5E5A", "verde": "#0F6E56", "morado": "#534AB7", "coral": "#993C1D"}


def _tarjeta(titulo, valor):
    return html.Div(
        [html.Div(titulo, style={"fontSize": "13px", "color": "#666"}),
         html.Div(valor, style={"fontSize": "26px", "fontWeight": "bold"})],
        style={"padding": "16px", "border": "1px solid #ddd", "borderRadius": "8px", "flex": "1", "textAlign": "center"},
    )


app.layout = html.Div([
    html.H1("Sistema de Recomendación Paralelo — RetailRocket", style={"marginBottom": "4px"}),
    html.P("Entrega 2 — Dashboard de integración (M1 + M2 + M3 + M4)", style={"color": "#666", "marginTop": "0"}),

    html.Div([
        _tarjeta("Eventos totales", f"{eventos.height:,}"),
        _tarjeta("Usuarios únicos", f"{segmentos.height:,}"),
        _tarjeta("Productos distintos", f"{eventos['itemid'].n_unique():,}"),
        _tarjeta("Clusters de usuario", f"{segmentos['cluster'].n_unique()}"),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "24px"}),

    dcc.Tabs(id="tabs", value="tab-usuarios", children=[
        dcc.Tab(label="Exploración de usuarios (M2)", value="tab-usuarios"),
        dcc.Tab(label="Interacciones y productos (M1)", value="tab-interacciones"),
        dcc.Tab(label="Recomendaciones (M3)", value="tab-recomendaciones"),
        dcc.Tab(label="Rendimiento (M4)", value="tab-rendimiento"),
        dcc.Tab(label="Prueba de carga (M5)", value="tab-carga"),
    ]),
    html.Div(id="contenido-tab", style={"marginTop": "20px"}),
])



# Callback: cambia el contenido según la pestaña activa

@app.callback(Output("contenido-tab", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-usuarios":
        fig_dist = px.bar(
            cluster_size, x="cluster_nombre", y="usuarios", text="usuarios",
            title="Distribución de usuarios por cluster",
            color_discrete_sequence=[COLORES["verde"]],
        )
        fig_dist.update_layout(xaxis_title="", yaxis_title="Usuarios", xaxis_tickangle=-20)

        # Cluster 2 es un outlier genuino: domina las 5 variables a la vez
        # (no solo una), así que normalizar 0-1 junto con los demás no
        # resuelve el problema — el resto sigue aplastado contra el centro.
        # Se separa en su propio gráfico y se normaliza el resto entre sí.
        columnas_perfil = ["Interacciones", "Productos distintos", "Views", "Carritos", "Compras"]
        es_outlier = perfil_clusters["cluster"] == 2

        perfil_resto = perfil_clusters[~es_outlier].copy()
        for col in columnas_perfil:
            minimo, maximo = perfil_resto[col].min(), perfil_resto[col].max()
            rango = maximo - minimo
            perfil_resto[col] = (perfil_resto[col] - minimo) / rango if rango > 0 else 0.0

        fig_perfil = go.Figure()
        for _, fila in perfil_resto.iterrows():
            fig_perfil.add_trace(go.Scatterpolar(
                r=[fila[c] for c in columnas_perfil], theta=columnas_perfil, fill="toself",
                name=NOMBRES_CLUSTER.get(int(fila["cluster"]), str(fila["cluster"])),
            ))
        fig_perfil.update_layout(
            title="Perfil — Clusters 0,1,3,4,5 (normalizado entre sí, sin el outlier)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        )

        fila_outlier = perfil_clusters[es_outlier].iloc[0]
        fig_outlier = go.Figure()
        fig_outlier.add_trace(go.Scatterpolar(
            r=[fila_outlier[c] for c in columnas_perfil], theta=columnas_perfil, fill="toself",
            line_color="#993C1D", name="Cluster 2",
        ))
        fig_outlier.update_layout(title="Perfil — Cluster 2 (Muy activo, escala propia)", polar=dict(radialaxis=dict(visible=True)))

        tabla_perfil = html.Table([
            html.Thead(html.Tr([html.Th("Cluster")] + [html.Th(c) for c in columnas_perfil])),
            html.Tbody([
                html.Tr(
                    [html.Td(NOMBRES_CLUSTER.get(int(fila["cluster"]), str(fila["cluster"])))]
                    + [html.Td(f"{fila[c]:.2f}") for c in columnas_perfil]
                )
                for _, fila in perfil_clusters.iterrows()
            ]),
        ], style={"width": "100%", "textAlign": "left", "marginTop": "12px"})

        return html.Div([
            dcc.Graph(figure=fig_dist),
            html.Div([dcc.Graph(figure=fig_perfil, style={"flex": "1"}), dcc.Graph(figure=fig_outlier, style={"flex": "1"})],
                     style={"display": "flex"}),
            html.H4("Valores absolutos por cluster"),
            tabla_perfil,
        ])

    elif tab == "tab-interacciones":
        fig_tipo = px.pie(
            eventos_por_tipo, names="event", values="cantidad",
            title="Distribución de tipos de evento",
            color_discrete_sequence=[COLORES["verde"], COLORES["coral"], COLORES["morado"]],
        )
        fig_hora = px.line(
            actividad_hora, x="hora_del_dia", y="interacciones", markers=True,
            title="Actividad por hora del día",
        )
        fig_top = px.bar(
            top_productos, x="itemid", y="interacciones", title="Top 15 productos con más interacciones",
            color_discrete_sequence=[COLORES["gris"]],
        )
        fig_top.update_layout(xaxis_title="ID de producto", xaxis_type="category")

        return html.Div([
            html.Div([dcc.Graph(figure=fig_tipo, style={"flex": "1"}), dcc.Graph(figure=fig_hora, style={"flex": "1"})],
                     style={"display": "flex"}),
            dcc.Graph(figure=fig_top),
        ])

    elif tab == "tab-recomendaciones":
        return html.Div([
            html.P("Selecciona un usuario (muestra de M3) para ver sus top-N recomendaciones del modelo ALS:"),
            dcc.Dropdown(
                id="selector-usuario",
                options=[{"label": f"Usuario {v}", "value": v} for v in usuarios_con_recomendaciones],
                value=usuarios_con_recomendaciones[0],
                style={"width": "300px"},
            ),
            html.Div(id="tabla-recomendaciones", style={"marginTop": "16px"}),
        ])

    elif tab == "tab-rendimiento":
        fig_resumen = px.bar(
            resumen_speedup, x="etapa", y="speedup_maximo", text="speedup_maximo",
            title="Speedup máximo alcanzado por etapa (M1 ETL, M2 Clustering, M3 Entrenamiento)",
            color_discrete_sequence=[COLORES["morado"]],
        )
        fig_resumen.update_traces(texttemplate="%{text:.2f}x", textposition="outside")

        fig_etl = go.Figure()
        fig_etl.add_trace(go.Scatter(x=speedup_etl["workers"], y=speedup_etl["speedup"], mode="lines+markers", name="Speedup real"))
        fig_etl.add_trace(go.Scatter(x=speedup_etl["workers"], y=speedup_etl["workers"], mode="lines", name="Speedup ideal", line=dict(dash="dash", color="gray")))
        fig_etl.update_layout(title="Speedup del ETL (M1) vs número de workers", xaxis_title="Workers", yaxis_title="Speedup")

        return html.Div([
            dcc.Graph(figure=fig_resumen),
            dcc.Graph(figure=fig_etl),
        ])

    elif tab == "tab-carga":
        if prueba_carga_df is None:
            return html.Div([
                html.P(
                    "Todavía no se ha corrido la prueba de carga. Corre "
                    "prueba_carga.py con el dashboard activo y guarda el resultado "
                    "en resultados/entrega3/prueba_carga.csv.",
                    style={"color": COLORES["alert"]},
                )
            ])

        fig_latencia = go.Figure()
        fig_latencia.add_trace(go.Scatter(
            x=prueba_carga_df["n_concurrentes"], y=prueba_carga_df["latencia_media_ms"],
            mode="lines+markers", name="Latencia media", line=dict(color=COLORES["coral"]),
        ))
        fig_latencia.add_trace(go.Scatter(
            x=prueba_carga_df["n_concurrentes"], y=prueba_carga_df["latencia_p95_ms"],
            mode="lines+markers", name="Latencia p95", line=dict(color=COLORES["morado"]),
        ))
        fig_latencia.update_layout(
            title="Latencia vs. usuarios concurrentes (simulación de pico de tráfico)",
            xaxis_title="Usuarios concurrentes", yaxis_title="Latencia (ms)",
        )

        fig_throughput = px.line(
            prueba_carga_df, x="n_concurrentes", y="throughput_req_por_s", markers=True,
            title="Throughput vs. usuarios concurrentes",
            color_discrete_sequence=[COLORES["verde"]],
        )
        fig_throughput.update_layout(xaxis_title="Usuarios concurrentes", yaxis_title="Solicitudes/segundo")
        fig_throughput.update_yaxes(range=[0, max(40, prueba_carga_df["throughput_req_por_s"].max() * 1.2)])

        tabla = html.Table([
            html.Thead(html.Tr([html.Th(c) for c in ["Concurrentes", "Latencia media", "Latencia p95", "Throughput", "Errores"]])),
            html.Tbody([
                html.Tr([
                    html.Td(int(r["n_concurrentes"])),
                    html.Td(f"{r['latencia_media_ms']:.0f} ms"),
                    html.Td(f"{r['latencia_p95_ms']:.0f} ms"),
                    html.Td(f"{r['throughput_req_por_s']:.1f} req/s"),
                    html.Td(f"{r['tasa_error']*100:.1f}%"),
                ]) for _, r in prueba_carga_df.iterrows()
            ]),
        ], style={"width": "100%", "textAlign": "left", "marginTop": "16px"})

        nota = html.P(
            "El throughput se mantiene aproximadamente constante sin importar la "
            "concurrencia -- el cuello de botella es el GIL de Python serializando "
            "el cómputo de cada respuesta (filtrado de datos + construcción de "
            "gráficos), no el modelo de conexiones del servidor.",
            style={"fontSize": "12px", "color": "#666", "marginTop": "12px"},
        )

        return html.Div([
            dcc.Graph(figure=fig_latencia),
            dcc.Graph(figure=fig_throughput),
            tabla,
            nota,
        ])


@app.callback(Output("tabla-recomendaciones", "children"), Input("selector-usuario", "value"))
def actualizar_recomendaciones(visitorid):
    subset = recomendaciones[recomendaciones["visitorid"] == visitorid].sort_values("rank")
    fig = px.bar(
        subset, x="rank", y="score", hover_data=["itemid"],
        title=f"Top-{len(subset)} recomendaciones para el usuario {visitorid}",
        color_discrete_sequence=["#0F6E56"],
    )
    # Se desactiva el formato SI automático de Plotly (que convertía valores
    # pequeños a notación "n"/"µ") y se fuerza notación científica legible.
    fig.update_layout(xaxis_title="Ranking", yaxis_title="Score del modelo ALS", yaxis_tickformat=".2e")
    tabla = html.Table([
        html.Thead(html.Tr([html.Th("Rank"), html.Th("Producto"), html.Th("Score")])),
        html.Tbody([
            html.Tr([html.Td(r["rank"]), html.Td(r["itemid"]), html.Td(f"{r['score']:.4e}")])
            for _, r in subset.iterrows()
        ]),
    ], style={"width": "100%", "textAlign": "left"})
    nota = html.P(
        "Nota: el score de ALS es un producto punto entre vectores latentes, no una "
        "probabilidad — lo relevante es el orden relativo, no la magnitud absoluta.",
        style={"fontSize": "12px", "color": "#666"},
    )
    return html.Div([dcc.Graph(figure=fig), tabla, nota])


if __name__ == "__main__":
    import os
    # En local (VS Code) corre igual que siempre: 127.0.0.1.
    # En Docker, docker-compose.yml define DASH_HOST=0.0.0.0 para que el
    # puerto expuesto sea accesible desde fuera del contenedor.
    host = os.environ.get("DASH_HOST", "127.0.0.1")
    debug = os.environ.get("DASH_DEBUG", "true").lower() == "true"
    app.run(host=host, port=8050, debug=debug, threaded=True)
