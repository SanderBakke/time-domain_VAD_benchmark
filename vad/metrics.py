from __future__ import annotations
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def clip_has_speech(frame_decisions: np.ndarray) -> int:
    """Return 1 if any frame is marked as speech, else 0."""
    return int(frame_decisions.max() > 0)

def evaluate_clip_level(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, a, b, c):
        self.dt = time.perf_counter() - self.t0
