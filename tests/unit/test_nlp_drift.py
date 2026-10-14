import numpy as np
import pandas as pd
import pytest

from xrayml.monitoring.drift import drift_report, psi
from xrayml.nlp.report_labeler import evaluate_labeler, label_report


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Conclusion : fracture non déplacée du radius distal.", "positive"),
        ("Conclusion : pas de fracture visible.", "negative"),
        ("Absence de trait de fracture.", "negative"),
        ("Fissure du scaphoïde à confirmer par scanner.", "uncertain"),
        ("Suspicion de fracture du col fémoral.", "uncertain"),
        ("IMPRESSION: No acute fracture or dislocation.", "negative"),
        ("Impression: Nondisplaced fracture of the fifth metatarsal.", "positive"),
        ("Fracture cannot be excluded.", "uncertain"),
        ("Fracture exclue.", "negative"),
        ("Une fracture ne peut être exclue.", "uncertain"),
        ("Pas de déplacement. Fracture de la styloïde ulnaire.", "positive"),
        ("Examen normal.", "not_mentioned"),
        # The findings section lists what was looked for; the conclusion wins.
        ("Résultats : recherche de fracture. Conclusion : examen normal.", "not_mentioned"),
    ],
)
def test_label_report_fr_en(text, expected):
    assert label_report(text, "fracture") == expected


def test_other_findings_and_agreement():
    assert label_report("Conclusion : épanchement articulaire du coude.", "effusion") == "positive"
    assert label_report("Pas de luxation.", "dislocation") == "negative"
    gold = [
        ("Pas de fracture.", "negative"),
        ("Fracture du radius.", "positive"),
        ("Possible fracture.", "uncertain"),
    ]
    assert evaluate_labeler(gold)["kappa"] == pytest.approx(1.0)


def test_psi_stable_vs_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.4, 0.1, 5000)
    assert psi(ref, rng.normal(0.4, 0.1, 5000)) < 0.02
    assert psi(ref, rng.normal(0.55, 0.1, 5000)) > 0.25  # new X-ray machine: brighter images


def test_drift_report_flags_the_drifting_site():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(
        {
            "mean_intensity": rng.normal(0.4, 0.1, 2000),
            "std_intensity": rng.normal(0.2, 0.02, 2000),
            "score": rng.random(2000),
            "positive": rng.random(2000) < 0.2,
        }
    )
    ok = ref.sample(300, random_state=1).assign(site="ok")
    bad = pd.DataFrame(
        {
            "mean_intensity": rng.normal(0.6, 0.1, 300),
            "std_intensity": rng.normal(0.2, 0.02, 300),
            "score": rng.random(300),
            "positive": rng.random(300) < 0.5,
            "site": "bad",
        }
    )
    rep = drift_report(ref, pd.concat([ok, bad])).set_index("site")
    assert not rep.loc["ok", "alert"] and rep.loc["bad", "alert"]
    assert "positive rate" in rep.loc["bad", "reason"] and "PSI(mean_intensity)" in rep.loc["bad", "reason"]
