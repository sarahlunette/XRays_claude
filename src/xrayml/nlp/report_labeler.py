"""Rule-based label mining from radiology reports, French and English (revision plan 4.4, 4.7).

The baseline every LLM/BERT labeller must beat. Pipeline: extract the conclusion section,
split sentences, find the finding, look for a negation cue *before* it within a window
(NegEx-style), then uncertainty cues. Output per report: positive / negative / uncertain /
not_mentioned. Validate on a radiologist-labelled gold set (per-class F1, Cohen's κ) and use
the output as *weak* labels for pretraining — never as test-set ground truth.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

Label = Literal["positive", "negative", "uncertain", "not_mentioned"]

FINDINGS = {
    "fracture": r"fractur\w*|fissur\w*|trait de fracture|break",
    "dislocation": r"luxation\w*|dislocat\w*",
    "effusion": r"epanchement\w*|effusion\w*",
}
NEGATION = r"\b(pas de|pas d|absence de|absence d|sans|aucun\w*|ni|no|not|without|negative for|free of|rule[sd]? out)\b"
POST_NEGATION = r"\b(exclue?s?|non retrouvee?s?|absente?s?|not seen|is excluded|none)\b"
UNCERTAIN = (
    r"\b(possible|probable|suspicion|suspect\w*|douteu\w*|a confirmer|cannot (be )?exclude[d]?|"
    r"cannot be ruled out|ne peut (pas )?etre exclue?|non exclue?|equivocal|may represent|evoqu\w*)\b"
)
SECTION = r"(conclusion|impression|interpretation)\s*:?"


def normalise(text: str) -> str:
    """Lowercase + strip accents so 'épanchement' and 'epanchement' match the same rule."""
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def conclusion(text: str) -> str:
    """Prefer the conclusion/impression section: findings sections list what was *looked for*."""
    t = normalise(text)
    m = re.search(SECTION, t)
    return t[m.end() :] if m else t


def label_report(text: str, finding: str = "fracture", window: int = 40) -> Label:
    pattern = FINDINGS[finding]
    found_any = False
    labels: list[Label] = []
    for sentence in re.split(r"[.;\n]+", conclusion(text)):
        for m in re.finditer(pattern, sentence):
            found_any = True
            before = sentence[max(0, m.start() - window) : m.start()]
            after = sentence[m.end() : m.end() + window]
            # order matters: "ne peut être exclue" is uncertain, "exclue" alone is negative
            if re.search(NEGATION, before):
                labels.append("negative")
            elif re.search(UNCERTAIN, sentence):
                labels.append("uncertain")
            elif re.search(POST_NEGATION, after):
                labels.append("negative")
            else:
                labels.append("positive")
    if not found_any:
        return "not_mentioned"
    # Any affirmed mention wins over a negated one ("no displacement, but a fracture of ...").
    for lab in ("positive", "uncertain", "negative"):
        if lab in labels:
            return lab
    return "not_mentioned"


def evaluate_labeler(gold: list[tuple[str, Label]], finding: str = "fracture") -> dict:
    from sklearn.metrics import classification_report, cohen_kappa_score

    y_true = [g for _, g in gold]
    y_pred = [label_report(t, finding) for t, _ in gold]
    return {
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
