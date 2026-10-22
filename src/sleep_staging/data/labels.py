import math

import pandas as pd

LABELS = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Movement time": "M",
}


def map_label(annotation: str, task: str) -> str | None:
    label = LABELS.get(annotation)
    return None if label == "M" and task == "task_5" else label


def epoch_annotations(
    annotations: list[tuple[float, float, str]], duration: float
) -> pd.DataFrame:
    rows = []
    for onset in range(0, math.ceil(duration), 30):
        intervals = [
            (start, start + length, text)
            for start, length, text in annotations
            if start < onset + 30 - 1e-6 and start + length > onset + 1e-6
        ]
        reason = ""
        original = None
        if onset + 30 > duration + 1e-6:
            reason = "incomplete"
        elif len(intervals) != 1:
            reason = "annotation_gap" if not intervals else "annotation_overlap"
        else:
            start, stop, original = intervals[0]
            if start > onset + 1e-6 or stop < onset + 30 - 1e-6:
                reason = "annotation_unaligned"
            elif original not in LABELS:
                reason = "unscored"
        rows.append(
            {
                "onset_s": onset,
                "onset_sample_eeg": onset * 100,
                "duration_s": min(30, duration - onset),
                "annotation": original,
                "task_5": map_label(original or "", "task_5") if not reason else None,
                "task_6": map_label(original or "", "task_6") if not reason else None,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)
