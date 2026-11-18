"""
GhostSignal ML module.

Public surface:
  - GhostSignalClassifier  : prediction interface
  - default_classifier     : shared instance for convenience imports
  - run_training           : train both pipelines
"""

from src.model.classifier import GhostSignalClassifier, default_classifier
from src.model.train import run_training

__all__ = ["GhostSignalClassifier", "default_classifier", "run_training"]
