"""
Grid segmentation module for multi-camera screen recordings.

Provides automatic grid detection, camera region extraction, and quality enhancement
for processing 100+ cameras in real-time from screen recordings.
"""
from .grid_detector import GridDetector
from .video_segmenter import VideoSegmenter
from .quality_enhancer import QualityEnhancer

__all__ = ['GridDetector', 'VideoSegmenter', 'QualityEnhancer']
