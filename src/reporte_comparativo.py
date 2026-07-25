"""
Genera la tabla comparativa final y las gráficas para la sección
'Modelado y Resultados' del informe IEEE.
"""
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados"


def cargar_json(nombre: str) -> dict:
    with open(RESULTADOS / nombre) as f:
        return json.load(f)


if __name__ == "__main__":
    comparacion = cargar_json("comparacion_als_vs_ncf.json")
    entrenamiento_neural = cargar_json("entrenamiento_neural.json")

    # --- Tabla comparativa ---
    tabla = pd.DataFrame({
        "Métrica": ["NDCG@10", "MAP@10", "Hit Rate@10", "Tiempo entrenamiento (s)", "Tiempo evaluación (s)"],
        "ALS (Baseline)": [
            comparacion["ALS"]["ndcg_at_10"],
            comparacion["ALS"]["map_at_10"],
            comparacion["ALS"]["hit_rate_at_10"],
            94.98,
            comparacion["ALS"]["tiempo_evaluacion_seg"],
        ],
        "NCF (Neural)": [
            comparacion["NCF"]["ndcg_at_10"],
            comparacion["NCF"]["map_at_10"],
            comparacion["NCF"]["hit_rate_at_10"],
            sum(entrenamiento_neural["tiempo_por_epoca_seg"]),
            comparacion["NCF"]["tiempo_evaluacion_seg"],
        ],
    })
    tabla.to_csv(RESULTADOS / "tabla_comparativa_final.csv", index=False)
    print("=== Tabla comparativa (guardada en resultados/tabla_comparativa_final.csv) ===")
    print(tabla.to_string(index=False))

    # --- Gráfica de métricas de calidad ---
    metricas = ["NDCG@10", "MAP@10", "Hit Rate@10"]
    valores_als = [comparacion["ALS"]["ndcg_at_10"], comparacion["ALS"]["map_at_10"], comparacion["ALS"]["hit_rate_at_10"]]
    valores_ncf = [comparacion["NCF"]["ndcg_at_10"], comparacion["NCF"]["map_at_10"], comparacion["NCF"]["hit_rate_at_10"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = range(len(metricas))
    ancho = 0.35
    axes[0].bar([i - ancho/2 for i in x], valores_als, ancho, label="ALS (Baseline)", color="#0F6E56")
    axes[0].bar([i + ancho/2 for i in x], valores_ncf, ancho, label="NCF (Neural)", color="#993C1D")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(metricas)
    axes[0].set_title("Calidad de recomendación (protocolo: 1 positivo + 99 negativos)")
    axes[0].legend()
    for i, (v1, v2) in enumerate(zip(valores_als, valores_ncf)):
        axes[0].text(i - ancho/2, v1 + 0.01, f"{v1:.3f}", ha="center", fontsize=9)
        axes[0].text(i + ancho/2, v2 + 0.01, f"{v2:.3f}", ha="center", fontsize=9)

    # --- Curva de pérdida (aunque sea 1 época) ---
    epocas = list(range(1, len(entrenamiento_neural["historial_perdida"]) + 1))
    axes[1].plot(epocas, entrenamiento_neural["historial_perdida"], marker="o", color="#993C1D")
    axes[1].set_title("Pérdida promedio por época — NCF")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("BCE Loss")
    axes[1].set_xticks(epocas)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0.693, color="gray", linestyle="--", linewidth=1, label="Predicción aleatoria (ln 2)")
    axes[1].legend()

    fig.suptitle("Comparación Baseline (ALS) vs. Modelo Neural (NCF) — Entrega 2", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "comparacion_als_vs_ncf.png", dpi=150)
    print(f"\nGráfica guardada en {RESULTADOS / 'comparacion_als_vs_ncf.png'}")