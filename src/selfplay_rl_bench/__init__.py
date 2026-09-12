"""Small reproducible adversarial self-play benchmark harness."""

from .trainer import TrainConfig, train
from .opponents import (
    RandomBot,
    WinBot,
    WinBlockBot,
    TacticalBot,
    NetOpponent,
    evaluate_vs_opponent,
    generate_tactical_probe,
    tactical_accuracy,
)

__all__ = [
    "TrainConfig",
    "train",
    "RandomBot",
    "WinBot",
    "WinBlockBot",
    "TacticalBot",
    "NetOpponent",
    "evaluate_vs_opponent",
    "generate_tactical_probe",
    "tactical_accuracy",
]
