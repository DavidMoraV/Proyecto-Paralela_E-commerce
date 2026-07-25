"""
Módulo de Machine Learning y Recomendación (M3).
Responsable: Siloé Campos

Contiene los modelos de recomendación del sistema:
- BaselineModel: filtrado colaborativo con ALS (implicit), para feedback implícito.
- NeuralModel: red neuronal (se agrega en el siguiente paso).
"""
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
import torch
import torch.nn as nn


class BaselineModel:
    """Filtrado colaborativo mediante ALS (Alternating Least Squares),
    apropiado para feedback implícito (vistas, carritos, compras con peso
    de confianza distinto, no ratings explícitos).
    """

    def __init__(self, factores: int = 64, regularizacion: float = 0.01,
                 iteraciones: int = 15, usar_gpu: bool = False):
        self.factores = factores
        self.regularizacion = regularizacion
        self.iteraciones = iteraciones
        self.usar_gpu = usar_gpu
        self.modelo = AlternatingLeastSquares(
            factors=factores,
            regularization=regularizacion,
            iterations=iteraciones,
            use_gpu=usar_gpu,
            random_state=42,
        )
        self.tiempo_entrenamiento_seg: float | None = None

    def entrenar(self, matriz_usuario_item: sp.csr_matrix) -> None:
        """Entrena el modelo. `implicit` espera la matriz en formato
        usuario x item, con los valores de confianza (peso_implicito)."""
        t0 = time.perf_counter()
        self.modelo.fit(matriz_usuario_item, show_progress=True)
        self.tiempo_entrenamiento_seg = time.perf_counter() - t0

    def recomendar(self, usuario_idx: int, matriz_usuario_item: sp.csr_matrix,
                    n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Top-N recomendaciones para un usuario. Devuelve (item_idx, scores)."""
        items, scores = self.modelo.recommend(
            usuario_idx,
            matriz_usuario_item[usuario_idx],
            N=n,
            filter_already_liked_items=True,
        )
        return items, scores

    def guardar(self, ruta: Path) -> None:
        self.modelo.save(str(ruta))

    @classmethod
    def cargar(cls, ruta: Path) -> "AlternatingLeastSquares":
        return AlternatingLeastSquares.load(str(ruta))

class NCFDataset(torch.utils.data.Dataset):
        """Envuelve los pares (usuario, item, etiqueta) para el DataLoader de PyTorch."""

        def __init__(self, usuarios: np.ndarray, items: np.ndarray, etiquetas: np.ndarray):
            self.usuarios = torch.as_tensor(usuarios, dtype=torch.long)
            self.items = torch.as_tensor(items, dtype=torch.long)
            self.etiquetas = torch.as_tensor(etiquetas, dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.usuarios)

        def __getitem__(self, idx: int):
            return self.usuarios[idx], self.items[idx], self.etiquetas[idx]


class NCFModelo(nn.Module):
        """Neural Collaborative Filtering: embeddings de usuario/item + MLP.
        Arquitectura basada en He et al. (2017), 'Neural Collaborative Filtering'.
        """

        def __init__(self, n_usuarios: int, n_items: int, dim_embedding: int = 32,
                     capas_ocultas: tuple[int, ...] = (64, 32, 16)):
            super().__init__()
            self.embedding_usuario = nn.Embedding(n_usuarios, dim_embedding)
            self.embedding_item = nn.Embedding(n_items, dim_embedding)

            capas = []
            dim_entrada = dim_embedding * 2
            for dim_salida in capas_ocultas:
                capas.append(nn.Linear(dim_entrada, dim_salida))
                capas.append(nn.ReLU())
                capas.append(nn.Dropout(0.2))
                dim_entrada = dim_salida
            capas.append(nn.Linear(dim_entrada, 1))
            self.mlp = nn.Sequential(*capas)

        def forward(self, usuario_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
            emb_u = self.embedding_usuario(usuario_idx)
            emb_i = self.embedding_item(item_idx)
            x = torch.cat([emb_u, emb_i], dim=1)
            logit = self.mlp(x).squeeze(-1)
            return logit  # sin sigmoid: se usa BCEWithLogitsLoss, más estable numéricamente


class NeuralModel:
        """Wrapper de entrenamiento/evaluación para NCFModelo, con detección
        automática de GPU (CUDA o MPS en Apple Silicon; CPU en su defecto)."""

        def __init__(self, n_usuarios: int, n_items: int, dim_embedding: int = 32,
                     capas_ocultas: tuple[int, ...] = (64, 32, 16), lr: float = 0.001):
            self.device = self._detectar_dispositivo()
            print(f"NeuralModel usando dispositivo: {self.device}")

            self.modelo = NCFModelo(n_usuarios, n_items, dim_embedding, capas_ocultas).to(self.device)
            self.optimizador = torch.optim.Adam(self.modelo.parameters(), lr=lr)
            self.criterio = nn.BCEWithLogitsLoss()
            self.historial_perdida: list[float] = []
            self.tiempo_por_epoca: list[float] = []

        @staticmethod
        def _detectar_dispositivo() -> torch.device:
            if torch.cuda.is_available():
                return torch.device("cuda")
            # MPS deshabilitado explícitamente: en Mac Intel da soporte inconsistente
            # con PyTorch 2.2.2 (cuelgues con operaciones de Embedding/int64).
            # Ver sección "Gestión de problemas" del informe.
            return torch.device("cpu")

        def entrenar(self, usuarios: np.ndarray, items: np.ndarray, etiquetas: np.ndarray,
                     n_epocas: int = 5, batch_size: int = 1024) -> None:
            import time
            from tqdm import tqdm

            dataset = NCFDataset(usuarios, items, etiquetas)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            self.modelo.train()
            for epoca in range(n_epocas):
                t0 = time.perf_counter()
                perdida_total = 0.0
                n_batches = 0

                barra = tqdm(loader, desc=f"Época {epoca + 1}/{n_epocas}")
                for lote_usuarios, lote_items, lote_etiquetas in barra:
                    lote_usuarios = lote_usuarios.to(self.device)
                    lote_items = lote_items.to(self.device)
                    lote_etiquetas = lote_etiquetas.to(self.device)

                    self.optimizador.zero_grad()
                    logits = self.modelo(lote_usuarios, lote_items)
                    perdida = self.criterio(logits, lote_etiquetas)
                    perdida.backward()
                    self.optimizador.step()

                    perdida_total += perdida.item()
                    n_batches += 1
                    barra.set_postfix(perdida=f"{perdida_total / n_batches:.4f}")

                perdida_promedio = perdida_total / n_batches
                tiempo_epoca = time.perf_counter() - t0
                self.historial_perdida.append(perdida_promedio)
                self.tiempo_por_epoca.append(tiempo_epoca)
                print(f"Época {epoca + 1}/{n_epocas} — pérdida: {perdida_promedio:.4f} "
                      f"— tiempo: {tiempo_epoca:.1f}s")

        def guardar(self, ruta: Path) -> None:
            torch.save(self.modelo.state_dict(), ruta)

if __name__ == "__main__":
    RAIZ = Path(__file__).resolve().parent.parent
    RUTA_MODELO_DATOS = RAIZ / "data" / "processed" / "modelo"
    RUTA_MODELOS_GUARDADOS = RAIZ / "resultados" / "modelos"
    RUTA_MODELOS_GUARDADOS.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("PARTE 1: BaselineModel (ALS)")
    print("=" * 60)
    matriz_train = sp.load_npz(RUTA_MODELO_DATOS / "matriz_train.npz")
    print(f"Matriz: {matriz_train.shape}, {matriz_train.nnz:,} valores no-cero")

    baseline = BaselineModel(factores=64, regularizacion=0.01, iteraciones=15)
    baseline.entrenar(matriz_train)
    print(f"Entrenamiento completo en {baseline.tiempo_entrenamiento_seg:.2f} segundos")
    baseline.guardar(RUTA_MODELOS_GUARDADOS / "baseline_als.npz")

    print("\n" + "=" * 60)
    print("PARTE 2: NeuralModel (NCF) — primer entrenamiento")
    print("=" * 60)
    datos_neural = np.load(RUTA_MODELO_DATOS / "pares_entrenamiento_neural.npz")
    usuarios, items, etiquetas = datos_neural["usuarios"], datos_neural["items"], datos_neural["etiquetas"]

    n_usuarios = matriz_train.shape[0]
    n_items = matriz_train.shape[1]
    print(f"n_usuarios: {n_usuarios:,} | n_items: {n_items:,} | pares de entrenamiento: {len(usuarios):,}")

    print(f"\nEntrenando NCF sobre el dataset completo ({len(usuarios):,} pares)...")
    neural = NeuralModel(n_usuarios, n_items, dim_embedding=32, capas_ocultas=(64, 32, 16))
    neural.entrenar(usuarios, items, etiquetas, n_epocas=5, batch_size=4096)

    neural.guardar(RUTA_MODELOS_GUARDADOS / "neural_ncf.pt")
    print(f"\nModelo neural guardado en {RUTA_MODELOS_GUARDADOS / 'neural_ncf.pt'}")

    import json
    resumen_neural = {
        "historial_perdida": neural.historial_perdida,
        "tiempo_por_epoca_seg": neural.tiempo_por_epoca,
        "dispositivo": str(neural.device),
    }
    with open(RAIZ / "resultados" / "entrenamiento_neural.json", "w") as f:
        json.dump(resumen_neural, f, indent=2)
    print("Resumen de entrenamiento guardado en resultados/entrenamiento_neural.json")
