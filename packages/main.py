# main.py
# Test całości: Moduły 1–4 na wybranym zbiorze (CSV)
import argparse
from pathlib import Path
import pandas as pd
import importlib.util
import sys


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="Test The Most Important Variables – Moduły 1–4")
    ap.add_argument("--data", type=str, required=True, help="Ścieżka do pliku CSV z danymi (np. avocado.csv)")
    ap.add_argument("--target", type=str, default=None, help="Nazwa kolumny docelowej (target), np. AveragePrice")
    ap.add_argument("--sample-n", type=int, default=0, help="Opcjonalne próbkowanie (np. 3000). 0 = bez próbkowania")
    ap.add_argument("--outdir", type=str, default="out", help="Katalog na wyniki")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {data_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Import modułów
    root = Path(__file__).parent
    m1 = _imp("target_det", root / "01_target_detection.py")
    m2 = _imp("model_train", root / "02_model_training.py")
    m3 = _imp("feature_report", root / "03_feature_report.py")
    m4 = _imp("final_report", root / "04_final_report.py")

    # Wczytaj dane
    df = pd.read_csv(data_path)
    if args.sample_n and args.sample_n > 0 and len(df) > args.sample_n:
        df = df.sample(n=args.sample_n, random_state=42).reset_index(drop=True)

    # Wybór targetu
    if args.target:
        target, typ, meta = m1.wybierz_target_i_typ(df, wybor_uzytkownika=args.target)
    else:
        target, typ, meta = m1.wybierz_target_i_typ(df)

    print(f"[INFO] Wybrany target: {target}  |  Typ problemu: {typ}  |  Źródło: {meta['zrodlo_wyboru']}")

    # Trening i ważności
    wynik = m2.trenowanie_automatyczne(df, target, typ, random_state=42)
    waznosci = wynik["waznosci"]
    metrics = wynik["metrics"]
    print("[INFO] Metryki:", metrics)

    # Wykres ważności
    png_bytes = m3.wykres_waznosci(waznosci, top_n=20, title=f"Ważność cech dla: {target}")
    png_path = outdir / f"feature_importance_{target}.png"
    png_path.write_bytes(png_bytes)

    # Rekomendacje
    rekom_md = m3.rekomendacje_tekstowe(typ, metrics, waznosci, max_punktow=5)
    md_path = outdir / f"feature_report_{target}.md"
    md_path.write_text(rekom_md, encoding="utf-8")

    # Raport HTML
    html_path = outdir / f"final_report_{target}.html"
    m4.zbuduj_raport_html(
        output_path=str(html_path),
        nazwa_projektu="The Most Important Variables",
        dataset_name=data_path.name,
        target=target,
        typ=typ,
        metrics=metrics,
        feature_png_path=str(png_path),
        waznosci_df=waznosci,
        rekomendacje_md=rekom_md,
        autor="AutoML – The Most Important Variables"
    )

    print(f"[DONE] Wyniki zapisane w: {outdir.resolve()}")
    print(f" - Wykres: {png_path.name}")
    print(f" - Rekomendacje: {md_path.name}")
    print(f" - Raport HTML: {html_path.name}")


if __name__ == "__main__":
    main()
