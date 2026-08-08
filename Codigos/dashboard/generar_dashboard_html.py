
from pathlib import Path

import pandas as pd
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio

# Rutas

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_PROCESSED = RAIZ / "datos"
CARPETA_RESULTADOS = RAIZ / "resultados"
SALIDA = Path(__file__).resolve().parent / "dashboard.html"

# Paleta / tokens de diseño

TOKENS = {
    "bg": "#0E1518",
    "surface": "#152024",
    "surface2": "#1B2A2F",
    "border": "#25373D",
    "text": "#EAF0EF",
    "text_muted": "#8FA5AA",
    "signal": "#34D9A6",   # exito / speedup / metricas positivas
    "warn": "#E8A33D",     # atencion / cuello de botella
    "alert": "#E1604F",    # problema encontrado
    "secondary": "#8C82F0",  # M3 / modelo
    "font_display": "'Space Grotesk', sans-serif",
    "font_body": "'Inter', sans-serif",
    "font_mono": "'IBM Plex Mono', monospace",
}

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=TOKENS["font_body"], color=TOKENS["text_muted"], size=13),
        title_font=dict(family=TOKENS["font_display"], color=TOKENS["text"], size=17),
        xaxis=dict(gridcolor=TOKENS["border"], zerolinecolor=TOKENS["border"], linecolor=TOKENS["border"]),
        yaxis=dict(gridcolor=TOKENS["border"], zerolinecolor=TOKENS["border"], linecolor=TOKENS["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=[TOKENS["signal"], TOKENS["secondary"], TOKENS["warn"], TOKENS["alert"], "#5FB8D9", "#C793E0"],
        margin=dict(t=50, l=50, r=30, b=50),
    )
)
pio.templates["m5_dark"] = PLOTLY_TEMPLATE
pio.templates.default = "m5_dark"


def fig_to_div(fig, div_id):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id, config={"displaylogo": False})



# Carga de datos

eventos = pl.read_parquet(CARPETA_PROCESSED / "eventos_limpios.parquet")
segmentos = pl.read_parquet(CARPETA_PROCESSED / "user_segments.parquet")
recomendaciones = pd.read_csv(CARPETA_RESULTADOS / "recomendaciones_muestra.csv")
resumen_speedup = pd.read_csv(CARPETA_RESULTADOS / "resumen_speedup_por_etapa.csv")
speedup_etl = pd.read_csv(CARPETA_RESULTADOS / "speedup_etl.csv")

NOMBRES_CLUSTER = {
    0: "Bajo/moderado", 1: "Ocasional (1 interacción)", 2: "Muy activo",
    3: "Sobre el promedio", 4: "Orientado a compra", 5: "Actividad intermedia",
}

eventos_por_tipo = eventos.group_by("event").agg(pl.len().alias("cantidad")).sort("cantidad", descending=True).to_pandas()
top_productos = (
    eventos.group_by("itemid").agg(pl.len().alias("interacciones"))
    .sort("interacciones", descending=True).head(15).to_pandas()
)
top_productos["itemid"] = top_productos["itemid"].astype(str)
actividad_hora = eventos.group_by("hora_del_dia").agg(pl.len().alias("interacciones")).sort("hora_del_dia").to_pandas()

cluster_size = segmentos.group_by("cluster").agg(pl.len().alias("usuarios")).sort("cluster").to_pandas()
cluster_size["nombre"] = cluster_size["cluster"].map(NOMBRES_CLUSTER)

perfil_clusters = (
    segmentos.group_by("cluster")
    .agg([
        pl.mean("total_interacciones").alias("Interacciones"),
        pl.mean("productos_distintos").alias("Productos distintos"),
        pl.mean("views").alias("Views"),
        pl.mean("carritos").alias("Carritos"),
        pl.mean("compras").alias("Compras"),
    ]).sort("cluster").to_pandas()
)

usuario_demo = int(recomendaciones["visitorid"].iloc[0])
top_rec_demo = recomendaciones[recomendaciones["visitorid"] == usuario_demo].sort_values("rank")


# Figuras

fig_cluster_dist = go.Figure(go.Bar(
    x=cluster_size["nombre"], y=cluster_size["usuarios"], marker_color=TOKENS["signal"],
    text=cluster_size["usuarios"], texttemplate="%{text:,}", textposition="outside",
))
fig_cluster_dist.update_layout(title="Distribución de usuarios por cluster", yaxis_title="Usuarios")

columnas_perfil = ["Interacciones", "Productos distintos", "Views", "Carritos", "Compras"]
es_outlier = perfil_clusters["cluster"] == 2
perfil_resto = perfil_clusters[~es_outlier].copy()
for col in columnas_perfil:
    mn, mx = perfil_resto[col].min(), perfil_resto[col].max()
    perfil_resto[col] = (perfil_resto[col] - mn) / (mx - mn) if mx > mn else 0.0

fig_radar = go.Figure()
for _, fila in perfil_resto.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[fila[c] for c in columnas_perfil], theta=columnas_perfil, fill="toself",
        name=NOMBRES_CLUSTER.get(int(fila["cluster"])),
    ))
fig_radar.update_layout(title="Perfil relativo — clusters 0,1,3,4,5", polar=dict(
    bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1], gridcolor=TOKENS["border"]),
    angularaxis=dict(gridcolor=TOKENS["border"]),
))

fig_tipo = go.Figure(go.Pie(
    labels=eventos_por_tipo["event"], values=eventos_por_tipo["cantidad"], hole=0.55,
    marker_colors=[TOKENS["signal"], TOKENS["warn"], TOKENS["alert"]],
))
fig_tipo.update_layout(title="Distribución de tipos de evento")

fig_hora = go.Figure(go.Scatter(
    x=actividad_hora["hora_del_dia"], y=actividad_hora["interacciones"],
    mode="lines+markers", line=dict(color=TOKENS["secondary"], width=3),
))
fig_hora.update_layout(title="Actividad por hora del día", xaxis_title="Hora", yaxis_title="Interacciones")

fig_top = go.Figure(go.Bar(x=top_productos["itemid"], y=top_productos["interacciones"], marker_color=TOKENS["text_muted"]))
fig_top.update_layout(title="Top 15 productos con más interacciones", xaxis_type="category", xaxis_title="ID de producto")

fig_rec = go.Figure(go.Bar(x=top_rec_demo["rank"], y=top_rec_demo["score"], marker_color=TOKENS["secondary"]))
fig_rec.update_layout(
    title=f"Top-{len(top_rec_demo)} recomendaciones — usuario {usuario_demo} (modelo ALS)",
    xaxis_title="Ranking", yaxis_title="Score", yaxis_tickformat=".2e",
)

fig_resumen = go.Figure(go.Bar(
    x=resumen_speedup["etapa"], y=resumen_speedup["speedup_maximo"], marker_color=TOKENS["warn"],
    text=resumen_speedup["speedup_maximo"], texttemplate="%{text:.2f}x", textposition="outside",
))
fig_resumen.update_layout(title="Speedup máximo por etapa del pipeline", yaxis_title="Speedup")

fig_etl_speedup = go.Figure()
fig_etl_speedup.add_trace(go.Scatter(x=speedup_etl["workers"], y=speedup_etl["speedup"], mode="lines+markers",
                                      name="Speedup real", line=dict(color=TOKENS["signal"], width=3)))
fig_etl_speedup.add_trace(go.Scatter(x=speedup_etl["workers"], y=speedup_etl["workers"], mode="lines",
                                      name="Speedup ideal", line=dict(color=TOKENS["text_muted"], dash="dash")))
fig_etl_speedup.update_layout(title="Speedup del ETL (M1) vs workers", xaxis_title="Workers", yaxis_title="Speedup")


# Ensamblar HTML

def tarjeta_kpi(valor, etiqueta):
    return f"""<div class="kpi"><div class="kpi-valor">{valor}</div><div class="kpi-etiqueta">{etiqueta}</div></div>"""


def nodo_pipeline(numero, nombre, activo=False):
    clase = "nodo activo" if activo else "nodo"
    return f"""<a href="#m{numero}" class="{clase}"><span class="nodo-punto"></span><span class="nodo-num">M{numero}</span><span class="nodo-nombre">{nombre}</span></a>"""


html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Sistema de Recomendación Paralelo — RetailRocket</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: {TOKENS['bg']}; --surface: {TOKENS['surface']}; --surface2: {TOKENS['surface2']};
    --border: {TOKENS['border']}; --text: {TOKENS['text']}; --text-muted: {TOKENS['text_muted']};
    --signal: {TOKENS['signal']}; --warn: {TOKENS['warn']}; --alert: {TOKENS['alert']}; --secondary: {TOKENS['secondary']};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: 'Inter', sans-serif; line-height: 1.5;
  }}
  h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; margin: 0; }}
  .mono {{ font-family: 'IBM Plex Mono', monospace; }}

  header {{ padding: 48px 56px 24px; border-bottom: 1px solid var(--border); }}
  .eyebrow {{ font-family: 'IBM Plex Mono', monospace; color: var(--signal); font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; }}
  header h1 {{ font-size: 34px; margin-top: 6px; }}
  header p {{ color: var(--text-muted); max-width: 640px; margin-top: 8px; }}

  .pipeline {{ display: flex; gap: 0; padding: 0 56px; margin-top: 28px; overflow-x: auto; }}
  .nodo {{ display: flex; align-items: center; gap: 8px; padding: 10px 18px; text-decoration: none; color: var(--text-muted); border-bottom: 2px solid var(--border); white-space: nowrap; transition: color .15s; }}
  .nodo:hover {{ color: var(--text); }}
  .nodo.activo {{ color: var(--text); border-bottom-color: var(--signal); }}
  .nodo.aqui {{ color: var(--bg); background: var(--signal); border-bottom-color: var(--signal); border-radius: 6px 6px 0 0; }}
  .nodo.aqui .nodo-num {{ color: var(--bg); font-weight: 700; }}
  .nodo.aqui .nodo-punto {{ background: var(--bg); box-shadow: none; }}
  .nodo-punto {{ width: 7px; height: 7px; border-radius: 50%; background: var(--signal); box-shadow: 0 0 8px var(--signal); }}
  .nodo-num {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--signal); }}
  .nodo-nombre {{ font-size: 13px; }}

  .kpis {{ display: flex; gap: 16px; padding: 32px 56px; flex-wrap: wrap; }}
  .kpi {{ flex: 1; min-width: 160px; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }}
  .kpi-valor {{ font-family: 'IBM Plex Mono', monospace; font-size: 28px; font-weight: 600; color: var(--signal); }}
  .kpi-etiqueta {{ color: var(--text-muted); font-size: 13px; margin-top: 4px; }}

  section {{ padding: 40px 56px; border-top: 1px solid var(--border); }}
  .section-head {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 24px; }}
  .section-head .tag {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--bg); background: var(--signal); padding: 3px 9px; border-radius: 4px; font-weight: 600; }}
  .section-head h2 {{ font-size: 22px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}

  select {{ background: var(--surface2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-family: 'Inter', sans-serif; margin-bottom: 16px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; color: var(--text-muted); font-weight: 500; padding: 8px; border-bottom: 1px solid var(--border); font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase; }}
  td {{ padding: 8px; border-bottom: 1px solid var(--border); }}

  footer {{ padding: 32px 56px; color: var(--text-muted); font-size: 12px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>

<header>
  <div class="eyebrow">Computación Paralela y Distribuida — Entrega 2</div>
  <h1>Sistema de Recomendación Paralelo</h1>
  <p>Pipeline completo sobre el dataset RetailRocket E-commerce: ingesta y limpieza distribuida, segmentación de usuarios, modelo de recomendación con feedback implícito, y análisis de rendimiento — con evidencia real de cada etapa.</p>
</header>

<nav class="pipeline">
  {nodo_pipeline(1, "Ingesta y ETL", True)}
  {nodo_pipeline(2, "Segmentación", True)}
  {nodo_pipeline(3, "Recomendación", True)}
  {nodo_pipeline(4, "Rendimiento", True)}
  <span class="nodo aqui"><span class="nodo-punto"></span><span class="nodo-num">M5</span><span class="nodo-nombre">Dashboard — estás aquí</span></span>
</nav>

<div class="kpis">
  {tarjeta_kpi(f"{eventos.height:,}", "Eventos procesados")}
  {tarjeta_kpi(f"{segmentos.height:,}", "Usuarios únicos")}
  {tarjeta_kpi(f"{eventos['itemid'].n_unique():,}", "Productos distintos")}
  {tarjeta_kpi(f"{segmentos['cluster'].n_unique()}", "Segmentos de usuario")}
</div>

<section id="m1">
  <div class="section-head"><span class="tag">M1</span><h2>Ingesta y preprocesamiento</h2></div>
  <div class="grid2">
    <div class="card">{fig_to_div(fig_tipo, "fig_tipo")}</div>
    <div class="card">{fig_to_div(fig_hora, "fig_hora")}</div>
  </div>
  <div class="card" style="margin-top:20px;">{fig_to_div(fig_top, "fig_top")}</div>
</section>

<section id="m2">
  <div class="section-head"><span class="tag">M2</span><h2>Segmentación de usuarios</h2></div>
  <div class="card">{fig_to_div(fig_cluster_dist, "fig_cluster_dist")}</div>
  <div class="grid2" style="margin-top:20px;">
    <div class="card">{fig_to_div(fig_radar, "fig_radar")}</div>
    <div class="card">
      <table>
        <thead><tr><th>Cluster</th><th>Interacciones</th><th>Productos</th><th>Views</th><th>Carritos</th><th>Compras</th></tr></thead>
        <tbody>
          {"".join(f"<tr><td>{NOMBRES_CLUSTER.get(int(r['cluster']))}</td><td class='mono'>{r['Interacciones']:.2f}</td><td class='mono'>{r['Productos distintos']:.2f}</td><td class='mono'>{r['Views']:.2f}</td><td class='mono'>{r['Carritos']:.2f}</td><td class='mono'>{r['Compras']:.2f}</td></tr>" for _, r in perfil_clusters.iterrows())}
        </tbody>
      </table>
    </div>
  </div>
</section>

<section id="m3">
  <div class="section-head"><span class="tag">M3</span><h2>Modelo de recomendación (ALS)</h2></div>
  <div class="card">{fig_to_div(fig_rec, "fig_rec")}</div>
  <p style="color:var(--text-muted); font-size:12px; margin-top:12px;">Usuario de ejemplo: {usuario_demo}. El score de ALS es un producto punto entre vectores latentes, no una probabilidad — lo relevante es el orden relativo, no la magnitud absoluta.</p>
</section>

<section id="m4">
  <div class="section-head"><span class="tag">M4</span><h2>Análisis de rendimiento</h2></div>
  <div class="grid2">
    <div class="card">{fig_to_div(fig_resumen, "fig_resumen")}</div>
    <div class="card">{fig_to_div(fig_etl_speedup, "fig_etl_speedup")}</div>
  </div>
</section>

<footer>Sistema de Recomendación Paralelo para E-Commerce — RetailRocket Dataset · Generado automáticamente a partir de los resultados de M1-M4.</footer>

</body>
</html>"""

SALIDA.write_text(html, encoding="utf-8")
print(f"Dashboard generado en {SALIDA} ({SALIDA.stat().st_size / 1024:.0f} KB)")
