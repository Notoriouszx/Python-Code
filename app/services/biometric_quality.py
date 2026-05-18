"""Layer 1 (image quality) and Layer 3 (liveness) checks before embedding extraction."""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from app.config import settings
from app.models.quality import ImageQualityReport


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _normalize_range(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp01((value - low) / (high - low))


class BiometricQualityChecker:
    """Assess capture quality and simple liveness cues (texture + color)."""

    def assess(self, bgr: np.ndarray, modality: str) -> ImageQualityReport:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(lap.var())

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        hf_texture = self._high_frequency_ratio(gray)
        color_variance = self._color_variance_score(bgr)

        sharp_score = _normalize_range(
            sharpness,
            settings.MIN_LAPLACIAN_VARIANCE * 0.5,
            settings.MIN_LAPLACIAN_VARIANCE * 3.0,
        )
        contrast_score = _normalize_range(contrast, 12.0, 55.0)
        brightness_score = 1.0
        if brightness < settings.MIN_BRIGHTNESS or brightness > settings.MAX_BRIGHTNESS:
            brightness_score = 0.0
        elif brightness < settings.MIN_BRIGHTNESS + 25 or brightness > settings.MAX_BRIGHTNESS - 25:
            brightness_score = 0.5

        hf_score = _normalize_range(
            hf_texture,
            settings.MIN_HF_TEXTURE_RATIO * 0.6,
            settings.MIN_HF_TEXTURE_RATIO * 2.5,
        )
        color_score = _normalize_range(
            color_variance,
            settings.MIN_COLOR_VARIANCE * 0.5,
            settings.MIN_COLOR_VARIANCE * 2.0,
        )

        liveness_score = _clamp01(0.45 * hf_score + 0.35 * color_score + 0.20 * sharp_score)

        if modality == "face":
            overall = _clamp01(
                0.25 * sharp_score
                + 0.20 * contrast_score
                + 0.15 * brightness_score
                + 0.20 * hf_score
                + 0.20 * color_score
            )
        else:
            overall = _clamp01(
                0.35 * sharp_score + 0.30 * contrast_score + 0.20 * brightness_score + 0.15 * hf_score
            )

        passed, reason = self._evaluate(
            modality=modality,
            sharpness=sharpness,
            contrast=contrast,
            brightness=brightness,
            hf_texture=hf_texture,
            color_variance=color_variance,
            overall=overall,
            liveness=liveness_score,
            min_side=min(h, w),
        )

        return ImageQualityReport(
            modality=modality,
            passed=passed,
            overall_score=overall,
            liveness_score=liveness_score,
            sharpness=sharpness,
            contrast=contrast,
            brightness=brightness,
            hf_texture=hf_texture,
            color_variance=color_variance,
            reject_reason=reason,
            details={
                "width": w,
                "height": h,
                "scores": {
                    "sharpness": round(sharp_score, 4),
                    "contrast": round(contrast_score, 4),
                    "brightness": round(brightness_score, 4),
                    "hf_texture": round(hf_score, 4),
                    "color_variance": round(color_score, 4),
                },
            },
        )

    def _evaluate(
        self,
        *,
        modality: str,
        sharpness: float,
        contrast: float,
        brightness: float,
        hf_texture: float,
        color_variance: float,
        overall: float,
        liveness: float,
        min_side: int,
    ) -> Tuple[bool, str | None]:
        if min_side < settings.MIN_IMAGE_SIDE:
            return False, f"Image too small (min {settings.MIN_IMAGE_SIDE}px on shortest side)"

        if brightness < settings.MIN_BRIGHTNESS or brightness > settings.MAX_BRIGHTNESS:
            return False, "Poor exposure (too dark or too bright)"

        if contrast < settings.MIN_CONTRAST:
            return False, "Low contrast — image appears flat or washed out"

        if sharpness < settings.MIN_LAPLACIAN_VARIANCE:
            return False, "Image too blurry (insufficient high-frequency detail)"

        if overall < settings.MIN_IMAGE_QUALITY_SCORE:
            return False, "Overall image quality below acceptable threshold"

        if modality == "face":
            if hf_texture < settings.MIN_HF_TEXTURE_RATIO:
                return False, "Missing fine skin texture (possible synthetic or over-smoothed image)"
            if color_variance < settings.MIN_COLOR_VARIANCE:
                return False, "Unnatural color distribution (possible AI-generated image)"
            if liveness < settings.MIN_LIVENESS_SCORE:
                return False, "Liveness check failed"

        return True, None

    @staticmethod
    def _high_frequency_ratio(gray: np.ndarray) -> float:
        """Share of spectral energy in the outer (high-frequency) band."""
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        total = float(np.sum(magnitude)) + 1e-8

        h, w = gray.shape
        cy, cx = h // 2, w // 2
        inner_r = max(4, int(min(h, w) * 0.12))
        y, x = np.ogrid[:h, :w]
        mask_inner = (y - cy) ** 2 + (x - cx) ** 2 <= inner_r**2
        hf_energy = float(np.sum(magnitude[~mask_inner]))
        return hf_energy / total

    @staticmethod
    def _color_variance_score(bgr: np.ndarray) -> float:
        """
        Real faces show channel imbalance and local chroma variation;
        synthetic renders are often overly uniform across B,G,R.
        """
        b, g, r = cv2.split(bgr.astype(np.float32))
        channel_stds = [float(np.std(c)) for c in (b, g, r)]
        mean_std = float(np.mean(channel_stds)) + 1e-8

        # Penalize when channels are nearly identical (low spread of stds).
        std_spread = float(np.std(channel_stds)) / mean_std

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        a = lab[:, :, 1].astype(np.float32)
        b_ch = lab[:, :, 2].astype(np.float32)
        chroma_std = float(np.std(a) + np.std(b_ch)) / 2.0

        return 0.55 * std_spread + 0.45 * (chroma_std / 64.0)
