from joblib import dump, load

from src.config import MODELS_DIR


def save_pipeline(pipeline, filename: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    dump(pipeline, path)
    print(f"Pipeline saved: {path.resolve()}")
    return path


def load_pipeline(filename: str):
    path = MODELS_DIR / filename
    pipeline = load(path)
    print(f"Pipeline loaded: {path.resolve()}")
    return pipeline