"""Versioned prompt files: `prompts/<version>/<name>.txt`.

Prompts are plain text files next to this module, so a wording change shows up as a
readable diff and a new version can be added without touching the old one.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load(name: str, version: str = "v1") -> str:
    """Return the prompt `name` of `version`.

    Raises FileNotFoundError when it does not exist: falling back to another prompt
    would silently change the behaviour of the model.
    """
    path = _PROMPTS_DIR / version / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()
