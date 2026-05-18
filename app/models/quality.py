from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ImageQualityReport:
    modality: str
    passed: bool
    overall_score: float
    liveness_score: float
    sharpness: float
    contrast: float
    brightness: float
    hf_texture: float
    color_variance: float
    reject_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "passed": self.passed,
            "overall_score": round(self.overall_score, 4),
            "liveness_score": round(self.liveness_score, 4),
            "sharpness": round(self.sharpness, 4),
            "contrast": round(self.contrast, 4),
            "brightness": round(self.brightness, 4),
            "hf_texture": round(self.hf_texture, 4),
            "color_variance": round(self.color_variance, 4),
            "reject_reason": self.reject_reason,
            "details": self.details,
        }


@dataclass
class EmbeddingQualityReport:
    modality: str
    passed: bool
    confidence: float
    peakiness: float
    entropy_ratio: float
    effective_dim_ratio: float
    reject_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "passed": self.passed,
            "confidence": round(self.confidence, 4),
            "peakiness": round(self.peakiness, 4),
            "entropy_ratio": round(self.entropy_ratio, 4),
            "effective_dim_ratio": round(self.effective_dim_ratio, 4),
            "reject_reason": self.reject_reason,
        }


@dataclass
class ModalityExtractionResult:
    embedding: Optional[Any] = None
    image_quality: Optional[ImageQualityReport] = None
    embedding_quality: Optional[EmbeddingQualityReport] = None
    rejected: bool = False
    reject_reason: Optional[str] = None

    @property
    def combined_quality(self) -> float:
        scores: list[float] = []
        if self.image_quality is not None:
            scores.append(self.image_quality.overall_score)
        if self.embedding_quality is not None:
            scores.append(self.embedding_quality.confidence)
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))
