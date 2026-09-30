"""Mantran: detect, erase, translate and re-letter manga speech bubbles."""

from .pipeline import process_image, translate_image

__all__ = ["process_image", "translate_image"]
