# app.py — The Most Important Variables (Streamlit)
# Test end-to-end: wybór targetu -> trening -> ważności -> raport HTML

import io
import sys
import base64
from pathlib import Path
import importlib.util

import streamlit as st
import pandas as pd

st.set_page_config(page_title="TMIV – The Most Important Variables", page_icon="📊", layout="wide")
st.title("📊 The Most Important Variables — Streamlit")

# ========= Pomocnicze: dynamiczny import modułu z pliku =========
def _imp(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)   # type: ignore
    return mod

# ========= Wczytanie modułów 01–04 (muszą leżeć obok app.py) =========
root = Path(__file__).parent
try:
    m1 = _imp("target_det", root / "01_target_detection.py")
    m2 = _imp("model_train", root / "02_model_training.py")
    m3 = _imp("feature_report", root / "03_feature_report.py")
    m4 = _imp("final_report", root / "04_final_report.py")
except Exception as e:
    st.error("Nie udało się załadować modułów 01–04. Upewnij się, że pliki **01_target_detection.py**, "
             "**02_model_training.py**, **03_feature_report.py**, **04_final_report.py** leżą obok `app.py`.")
    st.exception(e)
    st.stop()

# ========= Panel boczny: dane i ustawienia =========
with st.sidebar:
    st.header("⚙️ Ustawienia")
    up = st.file_uploader("Wgraj plik CSV", type=["csv"])
    use_sample = st.checkbox("Użyj pliku przykładowego (avocado.csv)", value=not bool(up))
    sample_n = st.number_input("Próbkowanie (0 = pełny zbiór)", min_value=0, value=0, step=500)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    st.markdown("---")
    run_btn = st.button("▶️ Uruchom analizę", type="primary", use_container_width=True)

# ========= Wczytanie danych =========
@st.cache_data(show_spinner=False)
def _read_csv(uploaded_file, use_sample_flag: bool) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if use_sample_flag:
        sample_path = root.parent.parent / "hackaton-09-08" / "data" / "avocado.csv"
        if sample_path.exists():
            return pd.read_csv(sample_path)
        else:
            raise FileNotFoundError("Zaznaczono przykład, ale **avocado.csv** nie znaleziono obok `app.py`.")
    raise ValueError("Nie wskazano źródła danych.")

try:
    df = _read_csv(up, use_sample)
    st.success(f"Wczytano dane: {len(df):,} wierszy × {len(df.columns)} kolumn")
    st.dataframe(df.head(20), use_container_width=True)
except Exception as e:
    st.info("Wgraj **CSV** lub zaznacz „Użyj pliku przykładowego (avocado.csv)” w panelu po lewej.")
    if up or use_sample:
        st.exception(e)
    st.stop()

# ========= Wybór targetu (ręczny vs auto) =========
st.subheader("🎯 Wybór zmiennej docelowej")
cols = st.columns([2, 1, 1, 1])
with cols[0]:
    manual_target = st.selectbox("Wybierz kolumnę (lub zostaw puste dla AUTO)", ["(AUTO)"] + list(df.columns))
with cols[1]:
    topN = st.number_input("Top N cech na wykresie", min_value=5, max_value=50, value=20, step=1)
with cols[2]:
    repeats = st.number_input("Permutation repeats", min_value=3, max_value=20, value=5, step=1)
with cols[3]:
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

# ========= Start analizy =========
if run_btn:
    try:
        # Próbkowanie (opcjonalnie)
        df_run = df
        if sample_n and sample_n > 0 and len(df) > sample_n:
            df_run = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

        # Wybor targetu i typu problemu
        if manual_target and manual_target != "(AUTO)":
            target, typ, meta = m1.wybierz_target_i_typ(df_run, wybor_uzytkownika=manual_target)
        else:
            target, typ, meta = m1.wybierz_target_i_typ(df_run)

        st.markdown(f"- **Wybrany target:** `{target}`  •  **Typ problemu:** **{typ}**  •  Źródło: `{meta['zrodlo_wyboru']}`")

        # Trening + ważności (użyjemy wersji z modułu 2; jeśli masz starszy sklearn, popraw w module 2 zgodnie z instrukcją)
        wynik = m2.trenowanie_automatyczne(df_run, target, typ, random_state=random_state)
        metrics = wynik["metrics"]
        waznosci = wynik["waznosci"]

        # Metryki
        st.subheader("📐 Metryki modelu")
        st.json(metrics)

        # Wykres ważności (PNG w pamięci) + pokaz w aplikacji
        st.subheader("📈 Ważność cech (Permutation Importance)")
        png_bytes = m3.wykres_waznosci(waznosci, top_n=int(topN), title=f"Ważność cech dla: {target}")
        st.image(png_bytes, caption="Feature importance", use_container_width=True)

        # Rekomendacje (Markdown)
        st.subheader("💡 Rekomendacje")
        md = m3.rekomendacje_tekstowe(typ, metrics, waznosci, max_punktow=5)
        st.markdown(md)

        # Przyciski pobierania: PNG i Raport HTML
        colA, colB = st.columns(2)
        with colA:
            st.download_button(
                "⬇️ Pobierz wykres (PNG)",
                data=png_bytes,
                file_name=f"feature_importance_{target}.png",
                mime="image/png",
                use_container_width=True,
            )
        with colB:
            # Zbuduj raport HTML z modułu 4 i podaj do pobrania
            html_path = root / f"final_report_{target}.html"
            # Zapisz tymczasowo PNG na dysk do osadzenia w raporcie
            tmp_png_path = root / f"_tmp_feature_{target}.png"
            tmp_png_path.write_bytes(png_bytes)
            html_file = m4.zbuduj_raport_html(
                output_path=str(html_path),
                nazwa_projektu="The Most Important Variables",
                dataset_name=up.name if up is not None else ("avocado.csv" if use_sample else "dataset.csv"),
                target=target,
                typ=typ,
                metrics=metrics,
                feature_png_path=str(tmp_png_path),
                waznosci_df=waznosci,
                rekomendacje_md=md,
                autor="AutoML – The Most Important Variables",
            )
            data = Path(html_file).read_bytes()
            st.download_button(
                "⬇️ Pobierz raport (HTML)",
                data=data,
                file_name=Path(html_file).name,
                mime="text/html",
                use_container_width=True,
            )
            # sprzątanie tymczasowego pliku PNG
            try:
                tmp_png_path.unlink(missing_ok=True)
            except Exception:
                pass

        # Podgląd tabeli ważności
        st.subheader("📋 Tabela ważności (Top 50)")
        st.dataframe(waznosci.head(50), use_container_width=True)

    except Exception as e:
        st.error("Coś poszło nie tak podczas analizy.")
        st.exception(e)
