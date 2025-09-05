from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Literal
import pandas as pd

import numpy as np
from pandas.api.types import (
    is_numeric_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_string_dtype,
)


SemanticType = Literal["numeric", "integer", "float", "boolean", "datetime", "categorical", "text", "unknown"]


@dataclass
class ColumnSchema:
    name: str
    pandas_dtype: str
    semantic_type: SemanticType
    n_unique: int
    unique_ratio: float
    is_unique: bool
    n_missing: int
    missing_ratio: float
    is_constant: bool
    example_non_null: Optional[Any] = None
    min_value: Optional[Any] = None           # dla numeric/datetime
    max_value: Optional[Any] = None           # dla numeric/datetime
    datetime_parse_rate: Optional[float] = None  # jeśli próbowaliśmy parsować obiekty na daty


@dataclass
class Schema:
    n_rows: int
    n_cols: int
    columns: Dict[str, ColumnSchema]
    primary_key_candidates: list[str]          # kolumny unikalne i bez braków
    notes: list[str]                           # luźne wskazówki, np. bardzo dużo braków


def _try_parse_datetime(series: pd.Series, sample_size: int = 100) -> tuple[float, Optional[pd.Series]]:
    """
    Próbnie parsuje wartości kolumny `object` jako daty.
    Zwraca odsetek poprawnie sparsowanych (na próbie) oraz pełną serię dat (lub None).
    """
    s = series.dropna()
    if s.empty:
        return 0.0, None

    sample = s.sample(min(sample_size, len(s)), random_state=42) if len(s) > sample_size else s
    # Próbuj różne formaty dat
    parsed_sample = None
    date_formats = [
        "%Y-%m-%d",           # 2025-01-01
        "%Y-%m-%d %H:%M:%S",  # 2025-01-01 12:00:00
        "%d/%m/%Y",           # 01/01/2025
        "%m/%d/%Y",           # 01/01/2025
        "%Y-%m-%dT%H:%M:%S",  # ISO format
        "%Y-%m-%dT%H:%M:%SZ", # ISO format z Z
        "%d-%m-%Y",           # 01-01-2025
        "%m-%d-%Y",           # 01-01-2025
    ]
    
    for date_format in date_formats:
        try:
            parsed_sample = pd.to_datetime(sample, format=date_format, errors="coerce", utc=True)
            if parsed_sample.notna().mean() >= 0.9:
                break
        except:
            continue
    
    # Jeśli żaden format nie zadziałał, spróbuj automatycznego parsowania z wyciszeniem ostrzeżeń
    if parsed_sample is None or parsed_sample.notna().mean() < 0.9:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed_sample = pd.to_datetime(sample, errors="coerce", utc=True)
    
    rate = float(parsed_sample.notna().mean())

    if rate >= 0.9:  # wysoki odsetek trafień -> traktujemy jako daty
        # Użyj tego samego formatu dla pełnej serii z wyciszeniem ostrzeżeń
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed_full = pd.to_datetime(series, errors="coerce", utc=True)
        return rate, parsed_full

    return rate, None


def _detect_semantic_type(s: pd.Series) -> tuple[SemanticType, Dict[str, Any]]:
    """
    Określa semantyczny typ kolumny (numeric/integer/float/boolean/datetime/categorical/text/unknown)
    oraz zwraca dodatkowe metryki (min/max, datetime_parse_rate).
    """
    info: Dict[str, Any] = {
        "min_value": None,
        "max_value": None,
        "datetime_parse_rate": None,
    }

    # 1) Natychmiastowe przypadki po dtype
    if is_bool_dtype(s):
        return "boolean", info

    if is_datetime64_any_dtype(s):
        if s.notna().any():
            min_val = s.min()
            max_val = s.max()
            # Konwertuj Timestamp na string dla kompatybilności z PyArrow
            info["min_value"] = str(min_val) if pd.notna(min_val) else None
            info["max_value"] = str(max_val) if pd.notna(max_val) else None
        return "datetime", info

    if is_numeric_dtype(s):
        # rozróżniamy integer vs float
        st: SemanticType = "integer" if is_integer_dtype(s) else ("float" if is_float_dtype(s) else "numeric")
        if s.notna().any():
            info["min_value"] = pd.to_numeric(s, errors="coerce").min()
            info["max_value"] = pd.to_numeric(s, errors="coerce").max()
        return st, info

    # 2) Dla object/string próbujemy:
    if is_string_dtype(s) or s.dtype == "object":
        # a) boolean-like?
        non_null = s.dropna().astype(str).str.strip().str.lower()
        boolean_tokens = {"true", "false", "t", "f", "yes", "no", "y", "n", "0", "1"}
        if not non_null.empty and non_null.isin(boolean_tokens).mean() >= 0.98:
            return "boolean", info

        # b) datetime-like?
        rate, parsed = _try_parse_datetime(s)
        info["datetime_parse_rate"] = rate
        if parsed is not None:
            if parsed.notna().any():
                min_val = parsed.min()
                max_val = parsed.max()
                # Konwertuj Timestamp na string dla kompatybilności z PyArrow
                info["min_value"] = str(min_val) if pd.notna(min_val) else None
                info["max_value"] = str(max_val) if pd.notna(max_val) else None
            return "datetime", info

        # c) categorical vs text (poziom unikalności)
        n_unique = s.nunique(dropna=True)
        n = len(s)
        if n > 0:
            ratio = n_unique / n
            # Heurystyka: mało unikalnych wzgl. liczby wierszy lub bezwzględnie niewiele kategorii
            if n_unique <= 50 or ratio <= 0.05:
                return "categorical", info
            return "text", info

    return "unknown", info


def infer_schema(df: pd.DataFrame) -> Schema:
    """
    Buduje schemat DataFrame:
      - pandas dtype i semantyczny typ kolumny,
      - liczność unikalnych wartości (i ratio),
      - brakujące wartości,
      - wykrywanie dat (także w kolumnach object),
      - flagi: unikalna/stała kolumna,
      - kandydaci na klucz główny.

    Nie modyfikuje wejściowego DataFrame.
    """
    n_rows, n_cols = df.shape
    cols: Dict[str, ColumnSchema] = {}
    notes: list[str] = []

    for name in df.columns:
        s = df[name]
        pandas_dtype = str(s.dtype)

        n_unique = int(s.nunique(dropna=True))
        unique_ratio = float(n_unique / len(s)) if len(s) else 0.0
        is_unique = bool((s.isna().sum() == 0) and (n_unique == len(s)))  # unikalne i bez braków
        n_missing = int(s.isna().sum())
        missing_ratio = float(n_missing / len(s)) if len(s) else 0.0
        is_constant = bool(n_unique == 1 or (n_unique == 0 and n_missing > 0))

        example_non_null = None
        non_null_values = s.dropna()
        if not non_null_values.empty:
            example_non_null = non_null_values.iloc[0]

        semantic_type, extra = _detect_semantic_type(s)

        col_schema = ColumnSchema(
            name=name,
            pandas_dtype=pandas_dtype,
            semantic_type=semantic_type,
            n_unique=n_unique,
            unique_ratio=unique_ratio,
            is_unique=is_unique,
            n_missing=n_missing,
            missing_ratio=missing_ratio,
            is_constant=is_constant,
            example_non_null=example_non_null,
            min_value=extra.get("min_value"),
            max_value=extra.get("max_value"),
            datetime_parse_rate=extra.get("datetime_parse_rate"),
        )
        cols[name] = col_schema

        # Luźne notatki pomocnicze
        if missing_ratio >= 0.3:
            notes.append(f"Kolumna '{name}' ma wysoki odsetek braków: {missing_ratio:.1%}.")
        if is_constant:
            notes.append(f"Kolumna '{name}' jest stała (brak zmienności).")
        if name.lower() in {"id", "uuid", "pk", "primary_key"} and not is_unique:
            notes.append(f"Kolumna '{name}' wygląda na identyfikator, ale nie jest unikalna.")

    primary_key_candidates = [
        c.name for c in cols.values() if c.is_unique and c.missing_ratio == 0.0
    ]

    return Schema(
        n_rows=n_rows,
        n_cols=n_cols,
        columns=cols,
        primary_key_candidates=primary_key_candidates,
        notes=notes,
    )

def schema_to_frame(schema: "Schema") -> pd.DataFrame:
    """
    Konwertuje Schema na tabelę z podsumowaniem kolumn.
    Kolumny: name, pandas_dtype, semantic_type, n_unique, unique_ratio, is_unique,
             n_missing, missing_ratio, is_constant, min_value, max_value,
             datetime_parse_rate, example_non_null
    """
    rows = []
    for col in schema.columns.values():
        rows.append({
            "name": col.name,
            "pandas_dtype": col.pandas_dtype,
            "semantic_type": col.semantic_type,
            "n_unique": col.n_unique,
            "unique_ratio": col.unique_ratio,
            "is_unique": col.is_unique,
            "n_missing": col.n_missing,
            "missing_ratio": col.missing_ratio,
            "is_constant": col.is_constant,
            "min_value": col.min_value,
            "max_value": col.max_value,
            "datetime_parse_rate": col.datetime_parse_rate,
            "example_non_null": col.example_non_null,
        })

    df_summary = pd.DataFrame(rows, columns=[
        "name", "pandas_dtype", "semantic_type",
        "n_unique", "unique_ratio", "is_unique",
        "n_missing", "missing_ratio", "is_constant",
        "min_value", "max_value", "datetime_parse_rate",
        "example_non_null"
    ])
    
    # Konwertuj min_value, max_value i example_non_null na stringi dla kompatybilności z PyArrow
    df_summary["min_value"] = df_summary["min_value"].astype(str)
    df_summary["max_value"] = df_summary["max_value"].astype(str)
    df_summary["example_non_null"] = df_summary["example_non_null"].astype(str)
    # Zamień 'None' z powrotem na None dla lepszej czytelności
    df_summary["min_value"] = df_summary["min_value"].replace("None", None)
    df_summary["max_value"] = df_summary["max_value"].replace("None", None)
    df_summary["example_non_null"] = df_summary["example_non_null"].replace("None", None)

    # Formatki procentów do czytelnego podglądu (nie zmieniają wartości liczbowych)
    # Jeśli wolisz surowe floaty, usuń te linie.
    df_summary["unique_ratio"] = df_summary["unique_ratio"].astype(float)
    df_summary["missing_ratio"] = df_summary["missing_ratio"].astype(float)

    return df_summary

# (opcjonalnie) helper do wygodnego podglądu jako dict (np. do JSON)
def schema_asdict(schema: Schema) -> dict:
    return {
        "n_rows": schema.n_rows,
        "n_cols": schema.n_cols,
        "columns": {k: asdict(v) for k, v in schema.columns.items()},
        "primary_key_candidates": schema.primary_key_candidates,
        "notes": list(schema.notes),
    }
