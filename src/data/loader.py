from __future__ import annotations
import pandas as pd
from pathlib import Path


def load_csv(path: str | Path, separator: str = ";", decimal: str = ",") -> pd.DataFrame:
    df = pd.read_csv(path, sep=separator, decimal=decimal)
    df.columns = df.columns.str.strip()
    return df


def load_excel(path: str | Path, sheet: int | str = 0) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


def load_from_config(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["path"])
    ext = path.suffix.lower()
    if ext == ".csv":
        return load_csv(path, cfg.get("separator", ";"), cfg.get("decimal", ","))
    elif ext in (".xls", ".xlsx"):
        return load_excel(path)
    raise ValueError(f"Desteklenmeyen dosya formatı: {ext}")
