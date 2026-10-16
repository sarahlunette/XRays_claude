import runpy
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "readme_terms_demo.py"


def test_readme_terms_demo_runs_and_shows_each_term(capsys):
    """docs/readme_explained.md quotes this script's output; keep it runnable."""
    runpy.run_path(str(SCRIPT), run_name="__main__")
    out = capsys.readouterr().out
    for marker in (
        "same split after shuffling the rows: True",
        "fitted T",
        "PPV",
        "flipped decisions",
        "PSI",
    ):
        assert marker in out
