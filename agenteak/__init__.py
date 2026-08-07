"""Agenteak: a multi-agent pipeline for terminology-constrained Spanish-Basque MT.

WMT26 Terminology Translation Task, Track 1 (es -> eu).
"""

__version__ = "1.0.0"

from .config import Config, MODE_SPECS, canonical_domain  # noqa: F401

__all__ = ["Config", "MODE_SPECS", "canonical_domain", "__version__"]
