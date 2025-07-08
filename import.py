from pathlib import Path

from types import FitPredictor


def find_repo_root(start_path: Path = Path.cwd()) -> Path:
    """Finds the root of the repo by locating the nearest pyproject.toml."""
    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("No pyproject.toml found in any parent directories.")


def get_single_file(path: Path) -> Path:
    files = [f for f in path.iterdir() if f.is_file()]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one file in {path}, found {len(files)}")
    return files[0]

def load_model() -> FitPredictor:
    repo_root = find_repo_root()
    model_path = get_single_file(repo_root / "model")
    raise NotImplementedError

