"""
Evaluación comparable de BaselineModel (ALS) y NeuralModel (NCF)
usando el protocolo estándar de NCF: leave-one-out con 99 negativos
muestreados por usuario (He et al., 2017).
"""
from pathlib import Path
import json
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from modelo_recomendacion import BaselineModel, NeuralModel, NCFModelo

N_NEGATIVOS_EVAL = 99
K = 10


def construir_vistos_por_usuario(train: pl.DataFrame) -> dict:
    vistos = {}
    for u, i in zip(train["usuario_idx"].to_numpy(), train["item_idx"].to_numpy()):
        vistos.setdefault(u, set()).add(i)
    return vistos


def muestrear_candidatos(test: pl.DataFrame, vistos_por_usuario: dict, n_items: int,
                          n_negativos: int = N_NEGATIVOS_EVAL, semilla: int = 42) -> list[tuple]:
    """Para cada usuario de test, genera (usuario, [item_verdadero + 99 negativos])."""
    rng = np.random.default_rng(semilla)
    casos = []
    for u, item_real in zip(test["usuario_idx"].to_numpy(), test["item_idx"].to_numpy()):
        vistos = vistos_por_usuario.get(u, set())
        negativos = []
        intentos = 0
        while len(negativos) < n_negativos and intentos < n_negativos * 20:
            candidato = rng.integers(0, n_items)
            intentos += 1
            if candidato not in vistos and candidato != item_real:
                negativos.append(candidato)
        candidatos = [item_real] + negativos  # posición 0 = el verdadero
        casos.append((u, candidatos))
    return casos


def calcular_metricas(rangos: list[int], k: int = K) -> dict:
    """rangos: posición (1-indexada) del ítem verdadero en el ranking de cada usuario."""
    ndcg, ap, aciertos = [], [], 0
    for r in rangos:
        if r <= k:
            ndcg.append(1.0 / np.log2(r + 1))
            ap.append(1.0 / r)
            aciertos += 1
        else:
            ndcg.append(0.0)
            ap.append(0.0)
    return {
        "ndcg_at_10": float(np.mean(ndcg)),
        "map_at_10": float(np.mean(ap)),
        "hit_rate_at_10": aciertos / len(rangos),
        "n_usuarios_evaluados": len(rangos),
    }


def evaluar_baseline_sampleado(baseline: BaselineModel, casos: list[tuple]) -> dict:
    """Rankea los 100 candidatos usando los factores de ALS ya entrenados."""
    factores_usuario = baseline.modelo.user_factors
    factores_item = baseline.modelo.item_factors

    rangos = []
    for u, candidatos in casos:
        scores = factores_item[candidatos] @ factores_usuario[u]
        orden = np.argsort(-scores)  # de mayor a menor score
        rango = int(np.where(orden == 0)[0][0]) + 1  # posición del item verdadero (índice 0)
        rangos.append(rango)
    return calcular_metricas(rangos)


def evaluar_neural_sampleado(neural: NeuralModel, casos: list[tuple], batch_size: int = 512) -> dict:
    neural.modelo.eval()
    rangos = []
    with torch.no_grad():
        for inicio in range(0, len(casos), batch_size):
            lote = casos[inicio:inicio + batch_size]
            usuarios_rep = []
            items_flat = []
            for u, candidatos in lote:
                usuarios_rep.extend([u] * len(candidatos))
                items_flat.extend(candidatos)

            t_usuarios = torch.tensor(usuarios_rep, dtype=torch.long, device=neural.device)
            t_items = torch.tensor(items_flat, dtype=torch.long, device=neural.device)
            logits = neural.modelo(t_usuarios, t_items).cpu().numpy()

            n_candidatos = len(lote[0][1])
            logits = logits.reshape(len(lote), n_candidatos)
            for fila in logits:
                orden = np.argsort(-fila)
                rango = int(np.where(orden == 0)[0][0]) + 1
                rangos.append(rango)
    return calcular_metricas(rangos)


if __name__ == "__main__":
    RAIZ = Path(__file__).resolve().parent.parent
    RUTA_MODELO_DATOS = RAIZ / "data" / "processed" / "modelo"
    RUTA_MODELOS_GUARDADOS = RAIZ / "resultados" / "modelos"

    print("Cargando datos...")
    train = pl.read_parquet(RUTA_MODELO_DATOS / "train.parquet")
    test = pl.read_parquet(RUTA_MODELO_DATOS / "test.parquet")
    matriz_train = sp.load_npz(RUTA_MODELO_DATOS / "matriz_train.npz")
    n_usuarios, n_items = matriz_train.shape

    print("Construyendo mapa de vistos por usuario...")
    vistos_por_usuario = construir_vistos_por_usuario(train)

    print(f"Muestreando {N_NEGATIVOS_EVAL} negativos por usuario de test...")
    t0 = time.perf_counter()
    casos = muestrear_candidatos(test, vistos_por_usuario, n_items)
    print(f"Listo en {time.perf_counter() - t0:.1f}s — {len(casos):,} casos de evaluación")

    print("\nCargando BaselineModel (ALS)...")
    baseline = BaselineModel()
    baseline.modelo = baseline.modelo.load(str(RUTA_MODELOS_GUARDADOS / "baseline_als.npz"))

    print("Evaluando ALS (protocolo sampleado)...")
    t0 = time.perf_counter()
    metricas_als = evaluar_baseline_sampleado(baseline, casos)
    metricas_als["tiempo_evaluacion_seg"] = time.perf_counter() - t0
    print(metricas_als)

    print("\nCargando NeuralModel (NCF)...")
    neural = NeuralModel(n_usuarios, n_items, dim_embedding=32, capas_ocultas=(64, 32, 16))
    neural.modelo.load_state_dict(torch.load(RUTA_MODELOS_GUARDADOS / "neural_ncf.pt"))

    print("Evaluando NCF (protocolo sampleado)...")
    t0 = time.perf_counter()
    metricas_ncf = evaluar_neural_sampleado(neural, casos)
    metricas_ncf["tiempo_evaluacion_seg"] = time.perf_counter() - t0
    print(metricas_ncf)

    resultado_final = {"ALS": metricas_als, "NCF": metricas_ncf}
    with open(RAIZ / "resultados" / "comparacion_als_vs_ncf.json", "w") as f:
        json.dump(resultado_final, f, indent=2)

    print("\n=== COMPARACIÓN FINAL (protocolo: 1 positivo + 99 negativos) ===")
    print(f"{'Métrica':<20} {'ALS':<12} {'NCF':<12}")
    for metrica in ["ndcg_at_10", "map_at_10", "hit_rate_at_10"]:
        print(f"{metrica:<20} {metricas_als[metrica]:<12.4f} {metricas_ncf[metrica]:<12.4f}")

    print(f"\nGuardado en resultados/comparacion_als_vs_ncf.json")