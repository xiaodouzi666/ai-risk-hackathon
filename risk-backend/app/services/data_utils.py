import pandas as pd, numpy as np, pathlib

# ---------- minimal helpers ---------------------------------
def load_dataset(task: str, protected_cols=None):
    """Return X, y, sens_df  (all list/array/DataFrame)."""
    if task == "image":
        df = pd.read_csv("data/img_test/metadata.csv")
        X = [pathlib.Path("data/img_test")/p for p in df["file"]]
    else:  # text
        df = pd.read_csv("data/nlp_test.csv")
        X = df["prompt"].tolist()
    y = df["label"].to_numpy()
    sens = (
        df[protected_cols] if protected_cols and
        set(protected_cols).issubset(df.columns)
        else pd.DataFrame()
    )
    return X, y, sens

def infer_attributes_with_clip(paths):
    """Dummy: mark all samples same group so code runs."""
    n = len(paths)
    return pd.DataFrame({"sex": ["na"]*n, "skin_tone": ["na"]*n})
