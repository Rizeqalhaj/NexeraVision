"""
Quality enhancement for low-resolution camera feeds.

Provides denoising, sharpening, contrast enhancement, and optional super-resolution.
"""
import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class QualityEnhancer:
    """
    Multi-stage quality enhancement pipeline for camera feeds.

    Enhances low-quality surveillance footage through denoising, sharpening,
    and contrast improvement.
    """

    def __init__(
        self,
        denoise_enabled: bool = True,
        sharpen_enabled: bool = True,
        contrast_enabled: bool = True,
        min_resolution: int = 240
    ):
        """
        Initialize quality enhancer.

        Args:
            denoise_enabled: Enable denoising step
            sharpen_enabled: Enable sharpening step
            contrast_enabled: Enable contrast enhancement
            min_resolution: Minimum resolution threshold for enhancement
        """
        self.denoise_enabled = denoise_enabled
        self.sharpen_enabled = sharpen_enabled
        self.contrast_enabled = contrast_enabled
        self.min_resolution = min_resolution

        # Sharpening kernel
        self.sharpen_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])

    def enhance_camera_feed(self, camera_frame: np.ndarray) -> np.ndarray:
        """
        Apply multi-stage enhancement pipeline to camera frame.

        Args:
            camera_frame: Input camera frame

        Returns:
            Enhanced camera frame
        """
        if camera_frame.size == 0:
            logger.warning("Empty camera frame, skipping enhancement")
            return camera_frame

        enhanced = camera_frame.copy()

        # 1. Denoise (remove compression artifacts)
        if self.denoise_enabled:
            enhanced = self._denoise(enhanced)

        # 2. Sharpen
        if self.sharpen_enabled:
            enhanced = self._sharpen(enhanced)

        # 3. Enhance contrast
        if self.contrast_enabled:
            enhanced = self._enhance_contrast(enhanced)

        return enhanced

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Remove noise and compression artifacts.

        Args:
            frame: Input frame

        Returns:
            Denoised frame
        """
        try:
            # Use fast non-local means denoising for color images
            denoised = cv2.fastNlMeansDenoisingColored(
                frame,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
            return denoised
        except Exception as e:
            logger.warning(f"Denoising failed: {e}, returning original")
            return frame

    def _sharpen(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter.

        Args:
            frame: Input frame

        Returns:
            Sharpened frame
        """
        try:
            sharpened = cv2.filter2D(frame, -1, self.sharpen_kernel)
            return sharpened
        except Exception as e:
            logger.warning(f"Sharpening failed: {e}, returning original")
            return frame

    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using histogram equalization in LAB color space.

        Args:
            frame: Input frame

        Returns:
            Contrast-enhanced frame
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)

            # Merge channels and convert back to BGR
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            return enhanced
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}, returning original")
            return frame

    def should_enhance(self, camera_frame: np.ndarray) -> bool:
        """
        Determine if frame needs enhancement based on resolution.

        Args:
            camera_frame: Camera frame to check

        Returns:
            True if enhancement is recommended
        """
        height, width = camera_frame.shape[:2]
        min_dimension = min(height, width)
        return min_dimension < self.min_resolution

    def apply_super_resolution(
        self,
        frame: np.ndarray,
        scale_factor: int = 2
    ) -> np.ndarray:
        """
        Apply basic super-resolution (placeholder for Real-ESRGAN).

        Note: This is a basic bicubic upscaling placeholder.
        For production, integrate Real-ESRGAN model.

        Args:
            frame: Input frame
            scale_factor: Upscaling factor (2x, 4x)

        Returns:
            Upscaled frame
        """
        height, width = frame.shape[:2]
        new_size = (width * scale_factor, height * scale_factor)

        # Use bicubic interpolation as placeholder
        upscaled = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)

        logger.info(f"Upscaled from {width}x{height} to {new_size[0]}x{new_size[1]}")
        return upscaled

    def batch_enhance(self, camera_feeds: list) -> list:
        """
        Enhance multiple camera feeds in batch.

        Args:
            camera_feeds: List of camera frame dictionaries

        Returns:
            List of enhanced camera frames
        """
        enhanced_feeds = []

        for i, camera in enumerate(camera_feeds):
            frame = camera['frame']

            # Only enhance if needed
            if self.should_enhance(frame):
                enhanced_frame = self.enhance_camera_feed(frame)
                camera['frame'] = enhanced_frame
                camera['enhanced'] = True
            else:
                camera['enhanced'] = False

            enhanced_feeds.append(camera)

        return enhanced_feeds


class AdaptiveEnhancer:
    """
    Adaptive quality enhancement that adjusts parameters based on frame quality.

    Analyzes frame quality metrics and applies appropriate enhancement level.
    """

    def __init__(self):
        """Initialize adaptive enhancer."""
        self.base_enhancer = QualityEnhancer()

    def analyze_quality(self, frame: np.ndarray) -> dict:
        """
        Analyze frame quality metrics.

        Args:
            frame: Input frame

        Returns:
            Dictionary with quality metrics:
                - sharpness: Laplacian variance (higher = sharper)
                - brightness: Mean pixel value
                - contrast: Standard deviation of pixel values
                - resolution: (width, height)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sharpness using Laplacian variance
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness
        brightness = np.mean(gray)

        # Contrast
        contrast = np.std(gray)

        # Resolution
        height, width = frame.shape[:2]

        return {
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'resolution': (width, height),
            'quality_score': self._calculate_quality_score(sharpness, brightness, contrast)
        }

    def _calculate_quality_score(
        self,
        sharpness: float,
        brightness: float,
        contrast: float
    ) -> float:
        """
        Calculate overall quality score (0-100).

        Args:
            sharpness: Sharpness metric
            brightness: Brightness metric
            contrast: Contrast metric

        Returns:
            Quality score (0-100, higher is better)
        """
        # Normalize metrics
        sharpness_score = min(sharpness / 1000, 1.0) * 40  # 40% weight
        brightness_score = (1.0 - abs(brightness - 128) / 128) * 30  # 30% weight (ideal ~128)
        contrast_score = min(contrast / 50, 1.0) * 30  # 30% weight

        total_score = sharpness_score + brightness_score + contrast_score
        return total_score

    def enhance(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Adaptively enhance frame based on quality analysis.

        Args:
            frame: Input frame

        Returns:
            Tuple of (enhanced_frame, quality_metrics)
        """
        # Analyze quality
        quality = self.analyze_quality(frame)

        # Adaptive enhancement based on quality score
        if quality['quality_score'] < 30:
            # Very low quality - aggressive enhancement
            enhancer = QualityEnhancer(
                denoise_enabled=True,
                sharpen_enabled=True,
                contrast_enabled=True
            )
            enhanced = enhancer.enhance_camera_feed(frame)
            quality['enhancement_level'] = 'aggressive'

        elif quality['quality_score'] < 60:
            # Medium quality - moderate enhancement
            enhancer = QualityEnhancer(
                denoise_enabled=True,
                sharpen_enabled=False,
                contrast_enabled=True
            )
            enhanced = enhancer.enhance_camera_feed(frame)
            quality['enhancement_level'] = 'moderate'

        else:
            # Good quality - minimal or no enhancement
            enhanced = frame
            quality['enhancement_level'] = 'none'

        # Re-analyze enhanced frame
        enhanced_quality = self.analyze_quality(enhanced)
        quality['enhanced_score'] = enhanced_quality['quality_score']
        quality['improvement'] = enhanced_quality['quality_score'] - quality['quality_score']

        return enhanced, quality
