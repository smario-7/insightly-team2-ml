# app.py - Streamlit aplikacja łącząca automatyczne wybieranie kolumny z trenowaniem modelu
import streamlit as st
import pandas as pd
import time
from pathlib import Path
import io
import base64

# Import z train.py
from train import train_model_with_auto_target, get_available_strategies

# Import z utils_v2
from utils_v2 import load_data, display_target_selection_with_spinner
from packages.schema_utils import infer_schema, schema_to_frame

# Import konfiguracji OpenAI
from config.settings import settings

# Konfiguracja strony
st.set_page_config(
    layout="wide",
    page_title="AutoML - The Most Important Variables",
    page_icon="🤖"
)

# Globalne style CSS dla lepszego wyświetlania
st.markdown("""
<style>
/* Agresywne style dla pełnej szerokości aplikacji */
.main .block-container {
    max-width: none !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Agresywne style dla komponentów HTML */
.stApp > div > div > div > div > div > div > div > div > iframe {
    width: 100vw !important;
    min-width: 100vw !important;
    max-width: 100vw !important;
    margin-left: -2rem !important;
    margin-right: -2rem !important;
}

/* Agresywne style dla tabel */
.stDataFrame {
    width: 100% !important;
    max-width: none !important;
}

/* Agresywne style dla wykresów */
.stPlotlyChart {
    width: 100% !important;
    max-width: none !important;
}

/* Agresywne style dla wszystkich kontenerów */
.stApp > div > div > div > div > div > div > div > div {
    max-width: none !important;
}

/* Agresywne style dla głównego kontenera */
.stApp > div > div > div > div > div > div > div {
    max-width: none !important;
}

/* Agresywne style dla elementów HTML */
.stApp > div > div > div > div > div > div > div > div > div > iframe {
    width: 100vw !important;
    min-width: 100vw !important;
    max-width: 100vw !important;
    margin-left: -2rem !important;
    margin-right: -2rem !important;
}

</style>
""", unsafe_allow_html=True)

# Inicjalizacja session state
if 'analysis_triggered' not in st.session_state:
    st.session_state.analysis_triggered = False
if 'last_analysis_params' not in st.session_state:
    st.session_state.last_analysis_params = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None

# Ścieżki
FOLDER = Path(__file__).resolve()
PATH = FOLDER.parent / "data" / "avocado.csv"  # data/avocado.csv

@st.cache_data(show_spinner=False)
def _read_csv_data(uploaded_file, use_default_flag: bool) -> pd.DataFrame:
    """Funkcja do wczytywania danych CSV z cache"""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if use_default_flag:
        default_path = FOLDER.parent / "data" / "avocado.csv"  # data/avocado.csv
        if default_path.exists():
            return pd.read_csv(default_path)
        else:
            raise FileNotFoundError(f"Nie znaleziono domyślnego pliku: {default_path}")
    raise ValueError("Nie wskazano źródła danych.")

def show_welcome_page():
    """Strona powitalna z wprowadzaniem klucza API"""
    st.title("🤖 AutoML - The Most Important Variables")
    st.markdown("*Automatyczna analiza danych z inteligentnym wyborem kolumny docelowej i trenowaniem modelu*")
    
    st.markdown("---")
    
    # Kolumny dla lepszego layoutu
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔑 Konfiguracja OpenAI API")
        st.markdown("""
        **Wprowadź swój klucz API OpenAI, aby korzystać z funkcji Auto AI:**
        - Automatyczny wybór kolumny docelowej przez AI
        - Inteligentna analiza danych
        - Zaawansowane rekomendacje
        """)
        
        # Pole do wprowadzania klucza API
        api_key = st.text_input(
            "Klucz API OpenAI:",
            type="password",
            placeholder="sk-...",
            help="Wprowadź swój klucz API OpenAI. Możesz go znaleźć na platform.openai.com"
        )
        
        # Przyciski
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Rozpocznij analizę", type="primary", use_container_width=True):
                if api_key:
                    # Zapisz klucz do session state
                    st.session_state.openai_api_key = api_key
                    st.session_state.show_main_app = True
                    st.rerun()
                else:
                    st.error("⚠️ Proszę wprowadzić klucz API OpenAI")
        
        with col_btn2:
            if st.button("⏭️ Kontynuuj bez AI", use_container_width=True):
                st.session_state.openai_api_key = ""
                st.session_state.show_main_app = True
                st.rerun()
    
    with col2:
        st.markdown("### 📋 Dostępne funkcje")
        
        if api_key.strip():
            st.success("✅ **Z kluczem API:**")
            st.markdown("""
            - 🤖 **Auto AI** - automatyczny wybór kolumny
            - 🧠 **Inteligentna analiza** - zaawansowane rekomendacje
            - 📊 **Pełna funkcjonalność** - wszystkie opcje
            """)
        else:
            st.info("ℹ️ **Bez klucza API:**")
            st.markdown("""
            - 🔍 **Heurystyka** - wybór kolumny na podstawie reguł
            - 📈 **Analiza danych** - podstawowe funkcje
            - 📋 **Raporty** - standardowe raporty
            """)
        
        st.markdown("### 💡 Wskazówki")
        st.markdown("""
        - Klucz API można wprowadzić później w ustawieniach
        - Bez klucza API nadal możesz korzystać z heurystyki
        - Wszystkie dane są przetwarzane lokalnie
        """)

def main():
    # Sprawdź czy użytkownik już wprowadził klucz API
    if 'show_main_app' not in st.session_state:
        st.session_state.show_main_app = False
    
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    
    # Jeśli nie pokazano jeszcze głównej aplikacji, pokaż stronę powitalną
    if not st.session_state.show_main_app:
        show_welcome_page()
        return
    
    # Główna aplikacja
    st.title("🤖 AutoML - The Most Important Variables")
    st.markdown("*Automatyczna analiza danych z inteligentnym wyborem kolumny docelowej i trenowaniem modelu*")

    # Wczytywanie danych - będzie zaktualizowane w sidebarze
    df = None
    data_loaded = False
    schema = None
    summary = None
    select_column_summary_schema = "missing_ratio"  # Domyślna wartość

    ### SIDEBAR ###
    with st.sidebar:
        st.header("⚙️ Ustawienia")
        
        # Status klucza OpenAI
        st.subheader("🔑 Status OpenAI")
        
        # Sprawdź czy klucz jest dostępny (tylko z session state)
        has_api_key = bool(st.session_state.openai_api_key)
        
        if has_api_key:
            st.success("✅ Klucz API OpenAI dostępny")
            st.info("🤖 Funkcja Auto AI będzie działać")
            st.markdown("""
            **💡 Wskazówki:**
            - Możesz przełączyć na strategię 'Heurystyka' jako alternatywa dla Auto AI
            """)
            
            # Przycisk do zmiany klucza
            if st.button("🔄 Zmień klucz API", use_container_width=True):
                st.session_state.show_main_app = False
                st.rerun()
        else:
            st.warning("⚠️ Brak klucza API OpenAI")
            st.info("🔍 Automatycznie przejdzie na heurystykę")
            st.markdown("""
            **Aby używać AI:**
            1. Kliknij "🔑 Wprowadź klucz API" poniżej
            2. Wprowadź swój klucz API OpenAI
            3. Klucz będzie przechowywany tylko w tej sesji
            """)
            
            # Przycisk do wprowadzenia klucza
            if st.button("🔑 Wprowadź klucz API", use_container_width=True):
                st.session_state.show_main_app = False
                st.rerun()
        
        st.markdown("---")
        
        # Ładowanie danych
        st.subheader("📁 Ładowanie danych")
        uploaded_file = st.file_uploader("Wgraj plik CSV", type=["csv"], help="Wybierz plik CSV do analizy")
        use_default_file = st.checkbox("Użyj domyślnego pliku (avocado.csv)", value=not bool(uploaded_file), help="Używa pliku hackaton-09-08/data/avocado.csv")
        
        st.markdown("---")
        
        # Wczytywanie danych na podstawie wyboru w sidebarze
        try:
            if uploaded_file is not None or use_default_file:
                df = _read_csv_data(uploaded_file, use_default_file)
                data_loaded = True
                
                # Analiza schematu
                schema = infer_schema(df)
                summary = schema_to_frame(schema)
                
        except Exception as e:
            st.error(f"❌ Błąd podczas wczytywania danych: {e}")
            if uploaded_file or use_default_file:
                st.exception(e)
            data_loaded = False
        
        if data_loaded:
            # Sortowanie podsumowania
            st.subheader("📊 Podsumowanie danych")
            select_column_summary_schema = st.selectbox(
                "Sortuj kolumny według:",
                options=summary.columns,
                index=summary.columns.get_loc(select_column_summary_schema) if select_column_summary_schema in summary.columns else 0,
                help="Wybierz metrykę do sortowania kolumn w podsumowaniu"
            )

            # Wybór strategii
            st.subheader("🎯 Strategia wyboru kolumny")
            strategies = get_available_strategies()
            
            # Filtruj dostępne strategie na podstawie klucza API
            if has_api_key:
                available_strategies = strategies
                default_index = 0  # auto_ai
            else:
                # Bez klucza API, usuń opcję auto_ai
                available_strategies = {k: v for k, v in strategies.items() if k != "auto_ai"}
                default_index = 0  # heuristics
            
            strategy_labels = list(available_strategies.keys())
            
            user_choice_label = st.selectbox(
                "Wybierz strategię:",
                options=strategy_labels,
                format_func=lambda x: available_strategies[x],
                index=default_index,
                key="strategy_selector",
                help="🤖 Auto AI - inteligentny wybór przez AI\n🔍 Heurystyka - analiza na podstawie nazw i typów\n👤 Ręczny - wybierz kolumnę ręcznie"
            )
            
            # Wybór kolumny dla strategii "manual"
            if user_choice_label == "manual":
                st.markdown("#### 👤 Wybór kolumny docelowej")
                st.info("Wybierz kolumnę, która będzie używana jako zmienna docelowa (target) w modelu ML")
                
                # Prosty selectbox z tylko nazwami kolumn
                selected_column = st.selectbox(
                    "Wybierz kolumnę docelową:",
                    options=df.columns.tolist(),
                    key="manual_column_selector",
                    help="Kolumna docelowa to ta, którą model będzie próbował przewidzieć"
                )
                
                # Zapisz wybór do session state
                st.session_state.manual_column_choice = selected_column
            else:
                # Wyczyść wybór jeśli nie jest to strategia manual
                if 'manual_column_choice' in st.session_state:
                    del st.session_state.manual_column_choice
            
            # Ustawienia ML
            st.subheader("🤖 Ustawienia ML")
            sample_n = st.number_input("Próbkowanie (0 = pełny zbiór)", min_value=0, value=0, step=500)
            random_state = st.number_input("Random state", min_value=0, value=42, step=1)
            top_n_features = st.number_input("Top N cech na wykresie", min_value=5, max_value=50, value=20, step=1)
            permutation_repeats = st.number_input("Permutation repeats", min_value=3, max_value=20, value=5, step=1)
            test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
            
            # Przycisk uruchomienia analizy
            run_analysis = st.button(
                "🚀 Uruchom analizę",
                type="primary",
                help="Kliknij aby uruchomić analizę według wybranej strategii"
            )
            
            # Logika uruchomienia analizy
            if run_analysis:
                # Sprawdź czy dla strategii manual wybrano kolumnę
                if user_choice_label == "manual" and 'manual_column_choice' not in st.session_state:
                    st.error("⚠️ Proszę wybrać kolumnę docelową dla strategii ręcznej")
                    st.stop()
                
                # Określ user_choice na podstawie wybranej strategii
                if user_choice_label == "auto_ai":
                    actual_user_choice = None
                elif user_choice_label == "heuristics":
                    actual_user_choice = "__force_heuristics__"
                elif user_choice_label == "manual":
                    actual_user_choice = st.session_state.manual_column_choice  # Użyj wybranej kolumny
                else:
                    actual_user_choice = "__force_manual__"
                
                # Zapisz parametry analizy
                analysis_params = {
                    'strategy_label': user_choice_label,
                    'user_choice': actual_user_choice,
                    'sample_n': sample_n,
                    'random_state': random_state,
                    'top_n_features': top_n_features,
                    'permutation_repeats': permutation_repeats,
                    'test_size': test_size
                }
                
                st.session_state.analysis_triggered = True
                st.session_state.last_analysis_params = analysis_params
                st.session_state.analysis_result = None
                st.session_state.ml_results = None
                
                strategies = get_available_strategies()
                st.success(f"🚀 Uruchamianie analizy: {strategies[user_choice_label]}")
                
            # Info o strategiach
            st.markdown("---")
            st.markdown("### 📝 Strategie wyboru:")
            strategies = get_available_strategies()
            for key, desc in strategies.items():
                if key == user_choice_label:
                    if key == "manual" and 'manual_column_choice' in st.session_state:
                        st.markdown(f"**{desc}** ✅ (Wybrano: `{st.session_state.manual_column_choice}`)")
                    else:
                        st.markdown(f"**{desc}** ✅")
                else:
                    st.markdown(f"{desc}")
            
            # Pokaż informację o niedostępnych strategiach
            strategies = get_available_strategies()
            if not has_api_key and "auto_ai" in strategies:
                st.markdown("---")
                st.info("ℹ️ **Auto AI** niedostępne - wprowadź klucz API OpenAI")
            
            st.markdown("**💡 Zmiana strategii nie uruchamia analizy!**")
            
        else:
            st.info("⏳ Załaduj poprawnie dane, aby wybrać kolumnę docelową")

    # Wyświetl informacje o załadowanych danych
    if data_loaded:
        if uploaded_file is not None:
            st.success(f"✅ Załadowano wgrane dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
            st.info(f"📁 Plik: {uploaded_file.name}")
        else:
            st.success(f"✅ Załadowano domyślne dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
            st.info("📁 Plik: avocado.csv")
    else:
        st.info("📁 **Wybierz plik CSV lub zaznacz domyślny plik w sidebarze**")

    ### TABS ###
    tab_summary, tab_selection, tab_ml, tab_results, tab_llm_report = st.tabs([
        "📊 Podsumowanie danych", 
        "🎯 Wybór targetu",
        "🤖 Trenowanie modelu",
        "📈 Wyniki i raporty",
        "⚛️ Raport z LLM", 

    ])

    with tab_summary:
        if data_loaded:
            st.markdown(f"### 📊 Analiza kolumn (sortowanie: **{select_column_summary_schema}**)")
            
            # Dodatkowe metryki
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Wiersze", f"{df.shape[0]:,}")
            with col2:
                st.metric("Kolumny", df.shape[1])
            with col3:
                missing_cols = sum(1 for col in schema.columns.values() if col.missing_ratio > 0.1)
                st.metric("Kolumny z brakami >10%", missing_cols)
            with col4:
                unique_cols = len(schema.primary_key_candidates)
                st.metric("Kandydaci na klucz", unique_cols)
            
            # Główne podsumowanie
            st.dataframe(
                summary.sort_values(select_column_summary_schema, ascending=False), 
                use_container_width=True,
                height=400
            )
            
            # Podgląd wybranej kolumny (jeśli strategia manual)
            if 'manual_column_choice' in st.session_state and st.session_state.manual_column_choice:
                st.markdown("---")
                st.markdown("### 👤 Podgląd wybranej kolumny docelowej")
                
                # Określ typ problemu ML
                selected_col = st.session_state.manual_column_choice
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    problem_type = "📊 Regresja"
                    model_type = "Regresja (przewidywanie wartości numerycznych)"
                else:
                    problem_type = "🏷️ Klasyfikacja"
                    model_type = "Klasyfikacja (przewidywanie kategorii)"
                
                # Metryki w kolumnach
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Typ danych", str(df[selected_col].dtype))
                    st.metric("Unikalne wartości", df[selected_col].nunique())
                
                with col2:
                    st.metric("Wartości brakujące", df[selected_col].isnull().sum())
                    st.metric("Procent braków", f"{(df[selected_col].isnull().sum() / len(df)) * 100:.1f}%")
                
                with col3:
                    if pd.api.types.is_numeric_dtype(df[selected_col]):
                        st.metric("Min", f"{df[selected_col].min():.2f}")
                        st.metric("Max", f"{df[selected_col].max():.2f}")
                    else:
                        st.metric("Najczęstsza wartość", df[selected_col].mode().iloc[0] if not df[selected_col].mode().empty else "Brak")
                
                with col4:
                    st.metric("Typ problemu", problem_type)
                    st.metric("Proponowany model", model_type)
            
            # Notatki
            if schema.notes:
                st.markdown("### ⚠️ Zauważone problemy:")
                for note in schema.notes:
                    st.warning(note)
                    
        else:
            st.warning("❌ Brak danych – nie można wyświetlić podsumowania")

    with tab_selection:
        st.markdown("## 🎯 Inteligentny wybór kolumny docelowej")
        
        if data_loaded:
            if st.session_state.get('analysis_triggered', False) and st.session_state.get('last_analysis_params'):
                params = st.session_state.last_analysis_params
                strategy_label = params['strategy_label']
                user_choice = params['user_choice']
                
                if st.session_state.analysis_result is None:
                    strategies = get_available_strategies()
                    st.info(f"🚀 Uruchamianie analizy: **{strategies[strategy_label]}**")
                    
                    # Analiza wyboru targetu
                    try:
                        decision = display_target_selection_with_spinner(
                            df, schema, user_choice, strategy_label, st.session_state.openai_api_key
                        )
                        
                        st.session_state.analysis_result = decision
                        st.session_state.analysis_triggered = False
                        
                    except Exception as e:
                        st.error(f"❌ Błąd podczas analizy: {e}")
                        if "429" in str(e) or "RateLimitError" in str(e) or "Too Many Requests" in str(e):
                            st.warning("⚠️ **Błąd limitu zapytań API**")
                            st.info("💡 Spróbuj ponownie za kilka minut lub przełącz na strategię 'Heurystyka'")
                        else:
                            st.exception(e)
                        st.session_state.analysis_triggered = False
                
                else:
                    st.success("✅ Wyniki analizy (zapisane):")
                    decision = st.session_state.analysis_result
                    
                    # Wyświetl wyniki ponownie
                    source_map = {
                        "user_choice": "🙋 Wybór użytkownika",
                        "llm_guess": "🤖 Propozycja AI", 
                        "heuristics_pick": "🔍 Analiza heurystyczna",
                        "none": "❌ Brak decyzji",
                    }
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Źródło decyzji", source_map.get(decision.source, decision.source))
                    
                    with col2:
                        if decision.target:
                            target_info = f"✅ {decision.target}"
                            if decision.target in schema.columns:
                                col_info = schema.columns[decision.target]
                                target_info += f" ({col_info.semantic_type})"
                        else:
                            target_info = "❌ Nie wybrano"
                        st.metric("Kolumna docelowa", target_info)
                    
                    if decision.reason:
                        if decision.source == "llm_guess":
                            st.success(f"🤖 **AI sugeruje**: {decision.reason}")
                        elif decision.source == "user_choice":
                            st.info(f"👤 **Kolumna wybrana przez użytkownika**")
                        elif decision.source == "heuristics_pick":
                            st.info(f"🔍 **Heurystyka**: {decision.reason}")
                        else:
                            st.error(f"❌ **Problem**: {decision.reason}")
            else:
                st.info("🎯 **Wybierz strategię w sidebarze i kliknij '🚀 Uruchom analizę' aby rozpocząć**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🤖 Auto AI")
                    st.markdown("""
                    - AI analizuje dane i proponuje najlepszą kolumnę
                    - Używa zaawansowanych algorytmów ML
                    - Pokazuje szczegółowe uzasadnienie
                    """)
                
                with col2:
                    st.markdown("### 🔍 Heurystyka")
                    st.markdown("""
                    - Analiza na podstawie nazw kolumn
                    - Ocena typów danych i jakości
                    - Szybkie wyniki bez AI
                    """)
                
                with col3:
                    st.markdown("### 👤 Ręczny wybór")
                    st.markdown("""
                    - Wybierz konkretną kolumnę z listy
                    - Natychmiastowe wyniki
                    - Pełna kontrola nad wyborem
                    """)
                    
        else:
            st.warning("❌ Brak danych – nie można uruchomić analizy")

    with tab_ml:
        st.markdown("## 🤖 Trenowanie modelu ML")
        
        if data_loaded and st.session_state.get('analysis_result'):
            decision = st.session_state.analysis_result
            
            if decision.target:
                st.success(f"🎯 **Wybrana kolumna docelowa**: {decision.target}")
                
                # Przycisk trenowania modelu
                if st.button("🚀 Trenuj model ML", type="primary"):
                    with st.spinner("🤖 Trenuję model ML..."):
                        try:
                            params = st.session_state.last_analysis_params
                            
                            # Uruchom trenowanie
                            result = train_model_with_auto_target(
                                df=df,
                                strategy=params['strategy_label'],
                                sample_n=params['sample_n'],
                                random_state=params['random_state'],
                                test_size=params['test_size'],
                                permutation_repeats=params['permutation_repeats'],
                                output_dir="out",
                                openai_api_key=st.session_state.openai_api_key
                            )
                            
                            st.session_state.ml_results = result
                            st.success("✅ Model wytrenowany pomyślnie!")
                            
                        except Exception as e:
                            st.error(f"❌ Błąd podczas trenowania: {e}")
                            st.exception(e)
                
                # Wyświetl wyniki ML jeśli są dostępne
                if st.session_state.get('ml_results'):
                    result = st.session_state.ml_results
                    
                    st.markdown("### 📊 Metryki modelu")
                    metrics = result['metrics']
                    
                    if result['type'] == 'regresja':
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", metrics.get('model', 'N/A'))
                        with col2:
                            st.metric("R²", f"{metrics.get('R2', 0):.3f}")
                        with col3:
                            st.metric("MAE", f"{metrics.get('MAE', 0):.3f}")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", metrics.get('model', 'N/A'))
                        with col2:
                            st.metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0):.3f}")
                        with col3:
                            st.metric("F1 Macro", f"{metrics.get('f1_macro', 0):.3f}")
                    
                    st.json(metrics)
                    
                    st.markdown("### 📈 Ważność cech")
                    feature_importance = result['feature_importance']
                    st.dataframe(feature_importance.head(20), use_container_width=True)
                    
                    st.markdown("### 💡 Rekomendacje")
                    st.markdown(result['recommendations'])
                    
            else:
                st.warning("❌ Najpierw wybierz kolumnę docelową w zakładce '🎯 Wybór targetu'")
        else:
            st.info("⏳ **Najpierw uruchom analizę wyboru kolumny docelowej**")

    with tab_results:
        st.markdown("## 📈 Wyniki i raporty")
        
        if st.session_state.get('ml_results'):
            result = st.session_state.ml_results
            
            st.markdown("### 📁 Pliki wygenerowane")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📊 Wykres ważności (PNG)")
                if Path(result['output_files']['png']).exists():
                    with open(result['output_files']['png'], 'rb') as f:
                        png_data = f.read()
                    st.download_button(
                        "⬇️ Pobierz wykres PNG",
                        data=png_data,
                        file_name=f"feature_importance_{result['target']}.png",
                        mime="image/png"
                    )
                else:
                    st.error("Plik PNG nie został wygenerowany")
            
            with col2:
                st.markdown("#### 📝 Rekomendacje (MD)")
                if Path(result['output_files']['md']).exists():
                    with open(result['output_files']['md'], 'r', encoding='utf-8') as f:
                        md_content = f.read()
                    st.download_button(
                        "⬇️ Pobierz rekomendacje MD",
                        data=md_content,
                        file_name=f"feature_report_{result['target']}.md",
                        mime="text/markdown"
                    )
                else:
                    st.error("Plik MD nie został wygenerowany")
            
            with col3:
                st.markdown("#### 🌐 Raport HTML")
                if Path(result['output_files']['html']).exists():
                    with open(result['output_files']['html'], 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.download_button(
                        "⬇️ Pobierz raport HTML",
                        data=html_content,
                        file_name=f"final_report_{result['target']}.html",
                        mime="text/html"
                    )
                else:
                    st.error("Plik HTML nie został wygenerowany")
            
            # WAŻNE: Podgląd raportu HTML poza kolumnami - na pełnej szerokości
            if Path(result['output_files']['html']).exists():
                st.markdown("---")  # Separator wizualny
                st.markdown("### 🌐 Podgląd raportu HTML")
                
                with open(result['output_files']['html'], 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Poprawka szerokości - użyj pełnej szerokości ekranu
                st.markdown("""
                <style>
                /* Agresywne style dla pełnej szerokości raportu HTML */
                .stApp > div > div > div > div > div > div > div > div > iframe {
                    width: 100vw !important;
                    min-width: 100vw !important;
                    max-width: 100vw !important;
                    margin-left: -2rem !important;
                    margin-right: -2rem !important;
                }
                
                /* Agresywne style dla wszystkich kontenerów iframe */
                .stApp > div > div > div > div > div > div > div > div > div > iframe {
                    width: 100vw !important;
                    min-width: 100vw !important;
                    max-width: 100vw !important;
                    margin-left: -2rem !important;
                    margin-right: -2rem !important;
                }
                
                /* Agresywne style dla głównego kontenera */
                .stApp > div > div > div > div > div > div > div {
                    max-width: none !important;
                }
                
                /* Agresywne style dla wszystkich kontenerów */
                .stApp > div > div > div > div > div > div > div > div {
                    max-width: none !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Wyświetlenie HTML na pełnej szerokości
                st.components.v1.html(
                    html_content, 
                    height=800, 
                    scrolling=True,
                    width=1400)
                
        else:
            st.info("⏳ **Najpierw wytrenuj model ML w zakładce '🤖 Trenowanie modelu'**")

    with tab_llm_report:
        st.markdown("## ⚛️ Raport z LLM")
        
        if st.session_state.get('llm_report'):
            report = st.session_state.llm_report
            st.markdown(report)
        else:
            st.info("⏳ rozwiązanie w trakcie tworzenia...")

if __name__ == "__main__":
    main()
 