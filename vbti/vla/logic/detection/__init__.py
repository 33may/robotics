from .detect import DuckDetector, OnnxDuckDetector, create_detector, StudentDetector
from .phases import PHASE_NAMES, detect_phases, detect_phases_episode, process_phases_dataset
from .process_dataset import process_dataset

__all__ = [
    "DuckDetector",
    "OnnxDuckDetector",
    "create_detector",
    "StudentDetector",
    "process_dataset",
    "PHASE_NAMES",
    "detect_phases",
    "detect_phases_episode",
    "process_phases_dataset",
]
