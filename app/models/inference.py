import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import settings

_MODALITIES = ("face", "fingerprint", "iris")


def _default_fusion_weights() -> Dict[str, float]:
    return {
        "face": settings.FACE_WEIGHT,
        "fingerprint": settings.FINGERPRINT_WEIGHT,
        "iris": settings.IRIS_WEIGHT,
    }


def _normalize_fusion_weights(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return _default_fusion_weights()
    out: Dict[str, float] = {}
    for m in _MODALITIES:
        if m in raw:
            out[m] = float(raw[m])
    return out or _default_fusion_weights()


def _synthetic_bundle() -> Dict[str, Any]:
    """Minimal in-memory gallery for local dev when MODEL_PATH is missing."""
    rng = np.random.default_rng(42)
    n_id = 3
    n_feat = 512
    n_comp = 64
    gallery: Dict[str, np.ndarray] = {}
    pca_models: Dict[str, Dict[str, Any]] = {}
    for m in _MODALITIES:
        comp = rng.standard_normal((n_comp, n_feat)).astype(np.float64)
        comp /= np.linalg.norm(comp, axis=1, keepdims=True) + 1e-8
        mean = rng.standard_normal(n_feat).astype(np.float64)
        pca_models[m] = {
            "components": comp,
            "mean": mean,
            "n_components": n_comp,
        }
        raw = rng.standard_normal((n_id, n_feat))
        centered = raw - mean
        proj = centered @ comp.T
        proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
        gallery[m] = proj.astype(np.float64)
    labels = np.arange(n_id, dtype=np.int64)
    return {
        "gallery_data": {"embeddings": gallery, "labels": labels},
        "pca_models": pca_models,
        "fusion_weights": _default_fusion_weights(),
    }


class BiometricInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.gallery_embeddings: Dict[str, np.ndarray] | None = None
        self.gallery_labels: np.ndarray | None = None
        self.pca_models: Dict[str, Dict[str, Any]] | None = None
        self.fusion_weights: Dict[str, float] = {}
        self.using_synthetic: bool = False
        self.load_models()

    def load_models(self) -> None:
        path = self.model_path
        if not os.path.isfile(path):
            print(f"No model file at {path}; using synthetic gallery for development.")
            data = _synthetic_bundle()
            self.using_synthetic = True
        else:
            print(f"Loading models from {path}...")
            with open(path, "rb") as f:
                data = pickle.load(f)

        gd = data["gallery_data"]
        emb = gd["embeddings"]
        self.gallery_embeddings = {
            "face": np.asarray(emb["face"], dtype=np.float64),
            "fingerprint": np.asarray(emb["fingerprint"], dtype=np.float64),
            "iris": np.asarray(emb["iris"], dtype=np.float64),
        }
        self.gallery_labels = np.asarray(gd["labels"])

        self.pca_models = {}
        for modality in _MODALITIES:
            pca_data = data["pca_models"][modality]
            self.pca_models[modality] = {
                "components": np.asarray(pca_data["components"], dtype=np.float64),
                "mean": np.asarray(pca_data["mean"], dtype=np.float64),
                "n_components": int(pca_data["n_components"]),
            }

        self.fusion_weights = _normalize_fusion_weights(
            data.get("fusion_weights", _default_fusion_weights())
        )

        n = len(self.gallery_labels)
        print(f"Loaded {n} gallery identities (synthetic={self.using_synthetic})")
        for m in _MODALITIES:
            print(f"   {m} gallery shape: {self.gallery_embeddings[m].shape}")

    def apply_pca(self, embedding: np.ndarray, modality: str) -> np.ndarray:
        if self.pca_models is None or modality not in self.pca_models:
            raise KeyError(modality)
        pca = self.pca_models[modality]
        emb = np.asarray(embedding, dtype=np.float64).ravel()
        mean = pca["mean"].ravel()
        if emb.shape[0] != mean.shape[0]:
            raise ValueError(
                f"Embedding dim {emb.shape[0]} does not match PCA mean {mean.shape[0]} for {modality}"
            )
        centered = emb - mean
        comp = pca["components"]
        transformed = centered @ comp.T
        transformed = transformed / (np.linalg.norm(transformed) + 1e-8)
        return transformed

    def _prepare_query_embedding(
        self, modality: str, embedding: np.ndarray
    ) -> np.ndarray:
        """Map raw embedding to gallery space (PCA + L2) when PCA input dim matches."""
        if self.pca_models is None:
            return np.asarray(embedding, dtype=np.float64).ravel()
        mean = self.pca_models[modality]["mean"].ravel()
        if embedding.size == mean.size:
            return self.apply_pca(embedding, modality)
        q = np.asarray(embedding, dtype=np.float64).ravel()
        gdim = self.gallery_embeddings[modality].shape[1] if self.gallery_embeddings else None
        if gdim is not None and q.size == gdim:
            return q / (np.linalg.norm(q) + 1e-8)
        raise ValueError(
            f"Query embedding length {embedding.size} incompatible with PCA/gallery for {modality}"
        )

    def identify(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.gallery_embeddings is None or self.gallery_labels is None:
            raise RuntimeError("Models not loaded")

        scores: Dict[str, np.ndarray] = {}
        for modality, emb in embeddings.items():
            if modality not in self.gallery_embeddings:
                continue
            q = self._prepare_query_embedding(modality, emb)
            gallery_emb = self.gallery_embeddings[modality]
            sim = np.dot(gallery_emb, q) / (
                np.linalg.norm(gallery_emb, axis=1) * np.linalg.norm(q) + 1e-8
            )
            scores[modality] = sim.astype(np.float64)

        if not scores:
            return {
                "user_id": None,
                "confidence": 0.0,
                "scores": {},
                "all_scores": [],
                "error": "No overlapping modalities",
            }

        fused = np.zeros(len(self.gallery_labels), dtype=np.float64)
        weight_sum = 0.0
        for modality, weight in self.fusion_weights.items():
            if modality in scores:
                fused += float(weight) * scores[modality]
                weight_sum += float(weight)
        if weight_sum > 0:
            fused /= weight_sum

        best_idx = int(np.argmax(fused))
        best_score = float(fused[best_idx])
        return {
            "user_id": int(self.gallery_labels[best_idx] + 1),
            "confidence": best_score,
            "scores": {k: float(v[best_idx]) for k, v in scores.items()},
            "all_scores": fused.tolist(),
        }

    def verify(
        self, embeddings: Dict[str, np.ndarray], target_user_id: int
    ) -> Dict[str, Any]:
        if self.gallery_embeddings is None or self.gallery_labels is None:
            raise RuntimeError("Models not loaded")

        label = target_user_id - 1
        hits = np.where(self.gallery_labels == label)[0]
        if len(hits) == 0:
            return {
                "verified": False,
                "confidence": 0.0,
                "scores": {},
                "error": "User not found",
            }
        target_idx = int(hits[0])

        scores: Dict[str, float] = {}
        for modality, emb in embeddings.items():
            if modality not in self.gallery_embeddings:
                continue
            q = self._prepare_query_embedding(modality, emb)
            gallery_emb = self.gallery_embeddings[modality][target_idx]
            sim = float(
                np.dot(q, gallery_emb)
                / (np.linalg.norm(q) * np.linalg.norm(gallery_emb) + 1e-8)
            )
            scores[modality] = sim

        fused_score = 0.0
        wsum = 0.0
        for modality, weight in self.fusion_weights.items():
            if modality in scores:
                fused_score += float(weight) * scores[modality]
                wsum += float(weight)
        if wsum > 0:
            fused_score /= wsum

        return {
            "verified": fused_score >= settings.VERIFICATION_THRESHOLD,
            "confidence": fused_score,
            "scores": scores,
        }


inference_model: Optional[BiometricInference] = None


def get_inference_model() -> BiometricInference:
    global inference_model
    if inference_model is None:
        inference_model = BiometricInference(settings.MODEL_PATH)
    return inference_model


def reset_inference_model_for_tests() -> None:
    global inference_model
    inference_model = None
