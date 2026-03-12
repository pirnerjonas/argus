"""Format detection diagnostics for near-miss datasets."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FormatProbe:
    """Diagnostic result for a dataset that partially matches a format.

    Attributes:
        path: Directory that was probed.
        format: Format name ("yolo", "coco", "mask").
        evidence_found: List of evidence strings describing what was found.
        evidence_missing: List of evidence strings describing what's missing.
        confidence: "strong" or "weak" indicating how close the match is.
    """

    path: Path
    format: str
    evidence_found: list[str] = field(default_factory=list)
    evidence_missing: list[str] = field(default_factory=list)
    confidence: str = "weak"

    @property
    def message(self) -> str:
        """Single-line diagnostic message for display."""
        found = "; ".join(self.evidence_found)
        missing = "; ".join(self.evidence_missing)
        return f"looks like {self.format.upper()} ({found}) but {missing}"
