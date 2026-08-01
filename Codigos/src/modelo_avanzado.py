"""
modelo_avanzado.py
Módulo 3 (M3) — Modelo avanzado de recomendación (Entrega 3)
Sistema de Recomendación Paralelo para E-Commerce — RetailRocket Dataset

Implementa NeuMF (Neural Matrix Factorization / NCF), combinando un camino
de factorización matricial generalizada (GMF) con un perceptrón multicapa
(MLP), siguiendo la arquitectura de He et al. (2017), "Neural Collaborative
Filtering". Diseñado para entrenarse con GPU (Google Colab); el código
también corre en CPU (más lento) para validación de lógica.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("M3_modelo_avanzado")


# ---------------------------------------------------------------------------
# 1. Arquitectura NeuMF (GMF + MLP)
# ---------------------------------------------------------------------------
class NeuMF(nn.Module):
    """Neural Matrix Factorization (He et al., 2017).

    Combina dos caminos que aprenden representaciones distintas del mismo
    par usuario-producto:
    - GMF (Generalized Matrix Factorization): producto elemento-a-elemento
      de los embeddings, generalización del producto punto usado en ALS.
    - MLP: concatenación de embeddings seguida de capas densas, capaz de
      aprender interacciones no lineales que ALS no puede representar.

    La salida final combina ambos caminos y pasa por una sigmoide, dando
    un score en (0, 1) interpretable como probabilidad de interacción.
    """

    def __init__(
        self,
        n_usuarios: int,
        n_items: int,
        dim_gmf: int = 32,
        dim_mlp: int = 32,
        capas_mlp: tuple[int, ...] = (64, 32, 16),
    ):
        super().__init__()

        self.emb_usuario_gmf = nn.Embedding(n_usuarios, dim_gmf)
        self.emb_item_gmf = nn.Embedding(n_items, dim_gmf)

        self.emb_usuario_mlp = nn.Embedding(n_usuarios, dim_mlp)
        self.emb_item_mlp = nn.Embedding(n_items, dim_mlp)

        capas = []
        dim_entrada = dim_mlp * 2
        for dim_salida in capas_mlp:
            capas.append(nn.Linear(dim_entrada, dim_salida))
            capas.append(nn.ReLU())
            dim_entrada = dim_salida
        self.mlp = nn.Sequential(*capas)

        self.capa_final = nn.Linear(dim_gmf + capas_mlp[-1], 1)
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        for emb in [self.emb_usuario_gmf, self.emb_item_gmf, self.emb_usuario_mlp, self.emb_item_mlp]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, usuario_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        gmf = self.emb_usuario_gmf(usuario_idx) * self.emb_item_gmf(item_idx)

        mlp_in = torch.cat([self.emb_usuario_mlp(usuario_idx), self.emb_item_mlp(item_idx)], dim=-1)
        mlp_out = self.mlp(mlp_in)

        combinado = torch.cat([gmf, mlp_out], dim=-1)
        return torch.sigmoid(self.capa_final(combinado)).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. Dataset con muestreo negativo
# ---------------------------------------------------------------------------
class InteraccionesImplicitasDataset(Dataset):
    """Para cada interacción positiva (usuario, item) en train, genera
    `n_negativos` pares negativos por época (items con los que el usuario
    NO interactuó), siguiendo el protocolo estándar de entrenamiento para
    feedback implícito (He et al., 2017).
    """

    def __init__(self, pares_positivos: np.ndarray, n_usuarios: int, n_items: int,
                 items_por_usuario: dict[int, set[int]], n_negativos: int = 4):
        self.pares_positivos = pares_positivos
        self.n_usuarios = n_usuarios
        self.n_items = n_items
        self.items_por_usuario = items_por_usuario
        self.n_negativos = n_negativos
        self._generar_muestras()

    def _generar_muestras(self):
        rng = np.random.default_rng()
        usuarios, items, labels = [], [], []
        for u, i in self.pares_positivos:
            usuarios.append(u); items.append(i); labels.append(1.0)
            vistos = self.items_por_usuario.get(u, set())
            negativos_agregados = 0
            intentos = 0
            while negativos_agregados < self.n_negativos and intentos < self.n_negativos * 10:
                candidato = rng.integers(0, self.n_items)
                intentos += 1
                if candidato not in vistos:
                    usuarios.append(u); items.append(candidato); labels.append(0.0)
                    negativos_agregados += 1

        self.usuarios = torch.tensor(usuarios, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.usuarios[idx], self.items[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# 3. Entrenamiento
# ---------------------------------------------------------------------------
def entrenar_ncf(
    modelo: NeuMF,
    dataset: InteraccionesImplicitasDataset,
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 0.001,
    device: str | None = None,
) -> dict:
    """Entrena el modelo con Binary Cross-Entropy (clasificación positivo/
    negativo). Devuelve historial de pérdida y tiempo total de entrenamiento.

    NOTA DE RENDIMIENTO: se indexa directamente sobre los tensores del
    dataset (ya completos en memoria) en vez de usar `torch.utils.data.
    DataLoader`. El DataLoader estándar llama a `__getitem__` muestra por
    muestra en Python puro antes de armar cada lote — con datasets de
    millones de muestras (como el de este proyecto, ~11.7M tras el muestreo
    negativo), ese overhead de Python domina sobre el cómputo real en GPU.
    Indexando tensores completos con `torch.randperm` se evita ese overhead
    por completo, ya que el "shuffle" y el armado de lotes se hacen como
    operaciones vectorizadas (en GPU, si el dataset se mueve ahí).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Entrenando NeuMF en dispositivo: %s", device)
    modelo.to(device)

    # Mover el dataset completo al dispositivo una sola vez (evita transferencias
    # CPU->GPU repetidas por lote).
    usuarios = dataset.usuarios.to(device)
    items = dataset.items.to(device)
    labels = dataset.labels.to(device)
    n_muestras = usuarios.shape[0]

    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.BCELoss()

    historial = []
    t0 = time.perf_counter()
    for epoca in range(epochs):
        modelo.train()
        permutacion = torch.randperm(n_muestras, device=device)
        # Se acumula como tensor en GPU (sin .item()) para no forzar una
        # sincronización GPU->CPU en cada uno de los ~miles de batches por
        # época -- solo se sincroniza una vez al final de la época.
        perdida_total = torch.zeros(1, device=device)

        for inicio in range(0, n_muestras, batch_size):
            idx = permutacion[inicio: inicio + batch_size]
            u_batch, i_batch, l_batch = usuarios[idx], items[idx], labels[idx]

            optimizador.zero_grad()
            pred = modelo(u_batch, i_batch)
            perdida = criterio(pred, l_batch)
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.detach() * len(idx)

        perdida_promedio = (perdida_total / n_muestras).item()  # única sincronización de la época
        historial.append(perdida_promedio)
        logger.info("Época %d/%d — BCE loss: %.4f", epoca + 1, epochs, perdida_promedio)

    tiempo_total = time.perf_counter() - t0
    logger.info("Entrenamiento completado en %.2f segundos", tiempo_total)
    return {"historial_perdida": historial, "tiempo_segundos": tiempo_total, "device": device}


# ---------------------------------------------------------------------------
# 4. Evaluación muestreada (protocolo estándar He et al. 2017: 1 positivo + N negativos)
# ---------------------------------------------------------------------------
def evaluar_ncf_muestreado(
    modelo: NeuMF,
    test_pares: np.ndarray,
    items_por_usuario: dict[int, set[int]],
    n_items: int,
    k: int = 10,
    n_negativos: int = 99,
    device: str | None = None,
) -> dict[str, float]:
    """Evalúa Hit Rate@K, MAP@K y NDCG@K usando el protocolo estándar de
    evaluación muestreada: para cada usuario de test, se rankea 1 producto
    verdadero contra `n_negativos` productos aleatorios no vistos (en vez de
    todo el catálogo), siguiendo la metodología de He et al. (2017). Esto
    permite una evaluación tratable en tiempo/memoria para redes neuronales,
    a diferencia de la evaluación de catálogo completo usada para ALS.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    modelo.eval()

    rng = np.random.default_rng(42)
    hits, ap_scores, ndcg_scores = [], [], []

    with torch.no_grad():
        for u, item_real in test_pares:
            vistos = items_por_usuario.get(u, set())
            negativos = []
            while len(negativos) < n_negativos:
                cand = int(rng.integers(0, n_items))
                if cand not in vistos and cand != item_real:
                    negativos.append(cand)

            candidatos = negativos + [int(item_real)]
            usuarios_t = torch.full((len(candidatos),), u, dtype=torch.long, device=device)
            items_t = torch.tensor(candidatos, dtype=torch.long, device=device)

            scores = modelo(usuarios_t, items_t).cpu().numpy()
            ranking = np.argsort(-scores)  # índices ordenados de mayor a menor score
            posicion_real = int(np.where(ranking == len(candidatos) - 1)[0][0]) + 1  # 1-indexado

            if posicion_real <= k:
                hits.append(1)
                ap_scores.append(1.0 / posicion_real)
                ndcg_scores.append(1.0 / np.log2(posicion_real + 1))
            else:
                hits.append(0); ap_scores.append(0.0); ndcg_scores.append(0.0)

    n = len(hits)
    return {
        "usuarios_evaluados": n,
        f"hit_rate@{k}": float(np.mean(hits)) if n else 0.0,
        f"map@{k}": float(np.mean(ap_scores)) if n else 0.0,
        f"ndcg@{k}": float(np.mean(ndcg_scores)) if n else 0.0,
        "protocolo": f"muestreado (1 positivo + {n_negativos} negativos)",
    }


# ---------------------------------------------------------------------------
# 5. Exportación del modelo (ONNX)
# ---------------------------------------------------------------------------
def exportar_onnx(modelo: NeuMF, ruta_salida: str, device: str = "cpu") -> None:
    """Exporta el modelo entrenado a formato ONNX, para inferencia portable
    fuera de PyTorch (requisito de Entrega 3: 'modelo exportado en formato
    estándar').
    """
    modelo.to(device)
    modelo.eval()
    usuario_dummy = torch.tensor([0], dtype=torch.long, device=device)
    item_dummy = torch.tensor([0], dtype=torch.long, device=device)

    torch.onnx.export(
        modelo,
        (usuario_dummy, item_dummy),
        ruta_salida,
        input_names=["usuario_idx", "item_idx"],
        output_names=["score"],
        dynamic_axes={"usuario_idx": {0: "batch"}, "item_idx": {0: "batch"}, "score": {0: "batch"}},
        opset_version=18,
    )
    logger.info("Modelo exportado a %s", ruta_salida)
