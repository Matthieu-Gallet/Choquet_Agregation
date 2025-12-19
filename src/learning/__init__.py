"""
Learning module for Choquet aggregation and ensemble classifiers.
"""

from .choquet_learnable import ChoquetClassifier, ChoquetTnormClassifier, preprocess
from .train_ensemble import main as train_ensemble_main
from .train_aggregate import main as train_aggregate_main

__all__ = [
    'ChoquetClassifier',
    'ChoquetTnormClassifier',
    'preprocess',
    'train_ensemble_main',
    'train_aggregate_main'
]
