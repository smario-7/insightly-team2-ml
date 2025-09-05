# 03_feature_report.py
# ==============================================
#  Moduł 3: Wizualizacja ważności cech + rekomendacje
# ==============================================

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

def wykres_waznosci(waznosci_df: pd.DataFrame, top_n: int = 20, title: Optional[str] = None) -> bytes:
    """
    Rysuje poziomy wykres słupkowy najważniejszych cech.
    Zwraca bajty PNG (do zapisania jako plik).
    """
    df = waznosci_df.copy().head(top_n).iloc[::-1]  # odwrócenie kolejności do barh
    fig, ax = plt.subplots(figsize=(8, max(4, int(0.35*len(df)))))
    ax.barh(df["cecha"], df["waznosc_srednia"])
    ax.set_xlabel("Ważność (permutation importance)")
    ax.set_ylabel("Cecha")
    if title:
        ax.set_title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def rekomendacje_tekstowe(
    typ: str,
    metrics: Dict[str, Any],
    waznosci_df: pd.DataFrame,
    max_punktow: int = 5
) -> str:
    """
    Generuje krótkie rekomendacje biznesowo-analityczne
    na bazie typu problemu, metryk i najważniejszych cech.
    """
    linie = []
    linie.append("# Rekomendacje\n")
    linie.append(f"- Typ problemu: **{typ}**\n")

    # Metryki
    linie.append("## Metryki\n")
    if typ == "regresja":
        linie.append(f"- Model: **{metrics.get('model', '')}**\n")
        linie.append(f"- R²: **{metrics.get('R2', 0):.3f}**\n")
        linie.append(f"- MAE: **{metrics.get('MAE', 0):.3f}**\n")
        linie.append(f"- RMSE: **{metrics.get('RMSE', 0):.3f}**\n")
    else:
        linie.append(f"- Model: **{metrics.get('model', '')}**\n")
        linie.append(f"- Zbalansowana trafność: **{metrics.get('balanced_accuracy', 0):.3f}**\n")
        linie.append(f"- Trafność: **{metrics.get('accuracy', 0):.3f}**\n")
        linie.append(f"- F1 (macro): **{metrics.get('f1_macro', 0):.3f}**\n")

    # Najważniejsze cechy
    linie.append("\n## Najważniejsze cechy\n")
    for i, row in enumerate(waznosci_df.head(max_punktow).itertuples(index=False), 1):
        linie.append(f"{i}. **{getattr(row, 'cecha')}** — ważność: {getattr(row, 'waznosc_srednia'):.5f}\n")

    # Ogólne wskazówki
    linie.append("\n## Sugestie działań\n")
    if typ == "regresja":
        linie.append("- Skup się na poprawie jakości i kompletności cech z najwyższą ważnością.\n")
        linie.append("- Rozważ inżynierię cech (transformacje, interakcje, grupowanie wartości).\n")
        linie.append("- Sprawdź rozkład błędów i obszary, gdzie model radzi sobie gorzej.\n")
        linie.append("- Jeśli R² jest niskie, dodaj więcej danych lub przetestuj bardziej złożone modele.\n")
    else:
        linie.append("- Zbalansuj klasy (class_weight, oversampling) i monitoruj trafność zbalansowaną.\n")
        linie.append("- Sprawdź, czy nie ma wycieku danych (data leakage).\n")
        linie.append("- Zoptymalizuj kodowanie cech kategorycznych (np. target encoding).\n")
        linie.append("- Analizuj macierz konfuzji, aby zrozumieć trudne przypadki.\n")

    return "".join(linie)

