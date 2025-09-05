# 04_final_report.py
# ==============================================
#  Moduł 4: Raport końcowy (HTML)
#  Zbiera metryki, wykres ważności i rekomendacje w jeden dokument.
# ==============================================

from typing import Dict, Any, Optional
import pandas as pd
import html as _html
from pathlib import Path
import base64

# --- prosta konwersja Markdown -> HTML (minimalna, bez zewnętrznych zależności) ---

def _md_line_to_html(line: str) -> str:
    l = line.rstrip("\n")
    if l.startswith("### "):
        return f"<h3>{_html.escape(l[4:])}</h3>"
    if l.startswith("## "):
        return f"<h2>{_html.escape(l[3:])}</h2>"
    if l.startswith("# "):
        return f"<h1>{_html.escape(l[2:])}</h1>"
    if l.startswith("- "):
        return f"<li>{_html.escape(l[2:])}</li>"
    # pogrubienie **...** (bardzo proste)
    if l.count("**") >= 2:
        parts = l.split("**")
        # zamień naprzemiennie na <b>...</b>
        out = []
        bold = False
        for p in parts:
            if bold:
                out.append(f"<b>{_html.escape(p)}</b>")
            else:
                out.append(_html.escape(p))
            bold = not bold
        l = "".join(out)
        return f"<p>{l}</p>"
    return f"<p>{_html.escape(l)}</p>"

def markdown_to_html(md: str) -> str:
    lines = md.splitlines()
    html_lines = []
    in_list = False
    for line in lines:
        if line.startswith("- "):
            if not in_list:
                in_list = True
                html_lines.append("<ul>")
            html_lines.append(_md_line_to_html(line))
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(_md_line_to_html(line))
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines)

# --- tabela top cech jako HTML ---

def df_top_features_html(waznosci_df: pd.DataFrame, top_n: int = 15) -> str:
    df = waznosci_df.head(top_n).copy()
    df["waznosc_srednia"] = df["waznosc_srednia"].round(6)
    df["waznosc_std"] = df["waznosc_std"].round(6)
    df["cecha"] = df["cecha"].astype(str)
    rows = []
    rows.append("<table class='tbl'><thead><tr><th>#</th><th>Cecha</th><th>Ważność</th><th>Std</th></tr></thead><tbody>")
    for i, r in enumerate(df.itertuples(index=False), 1):
        rows.append(
            f"<tr><td>{i}</td>"
            f"<td>{_html.escape(getattr(r,'cecha'))}</td>"
            f"<td>{getattr(r,'waznosc_srednia')}</td>"
            f"<td>{getattr(r,'waznosc_std')}</td></tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)

# --- wklejenie PNG na stronę jako data URI ---

def img_to_data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/png;base64,{b64}"

# --- główna funkcja budująca raport ---

def zbuduj_raport_html(
    output_path: str,
    nazwa_projektu: str,
    dataset_name: str,
    target: str,
    typ: str,
    metrics: Dict[str, Any],
    feature_png_path: Optional[str],
    waznosci_df: pd.DataFrame,
    rekomendacje_md: str,
    autor: str = "AutoML – The Most Important Variables"
) -> str:
    """Buduje spójny raport HTML i zapisuje pod output_path. Zwraca ścieżkę do pliku."""
    # metryki – render
    metrics_html_items = []
    for k, v in metrics.items():
        metrics_html_items.append(f"<li><b>{_html.escape(str(k))}:</b> {_html.escape(str(v))}</li>")
    metrics_html = "<ul>" + "\n".join(metrics_html_items) + "</ul>"

    # obraz wykresu
    img_html = ""
    if feature_png_path:
        data_uri = img_to_data_uri(feature_png_path)
        if data_uri:
            img_html = (
                f"<img alt='Feature importance' src='{data_uri}' "
                f"style='max-width:100%;height:auto;border:1px solid #ddd;border-radius:10px;padding:8px;'/>"
            )

    # tabela top cech
    features_table = df_top_features_html(waznosci_df, top_n=15)

    # rekomendacje (markdown -> html)
    rekomendacje_html = markdown_to_html(rekomendacje_md)

    css = '''
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color: #0f172a; }
      header { display:flex; align-items:center; gap:16px; margin-bottom:16px; }
      .badge { background:#eef2ff; color:#3730a3; padding:4px 10px; border-radius:999px; font-size:12px; }
      h1 { margin: 0; font-size: 28px; }
      .meta { color:#334155; font-size:14px; margin-bottom:16px; }
      .card { background:#ffffff; border:1px solid #e5e7eb; border-radius:16px; padding:16px; margin: 16px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
      .tbl { width:100%; border-collapse: collapse; }
      .tbl th, .tbl td { border-bottom:1px solid #e5e7eb; text-align:left; padding:8px; font-size:14px; }
      .tbl th { background:#f8fafc; }
      footer { margin-top: 24px; color:#64748b; font-size:12px; }
      .pill { display:inline-block; border:1px solid #e5e7eb; padding:4px 10px; border-radius:999px; margin-right:8px; background:#f8fafc; }
    </style>
    '''

    html_doc = f"""
<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8"/>
<title>Raport – {_html.escape(nazwa_projektu)}</title>
{css}
</head>
<body>
  <header>
    <div class="badge">Raport</div>
    <h1>{_html.escape(nazwa_projektu)}</h1>
  </header>

  <div class="meta">
    <span class="pill">Zbiór: {_html.escape(dataset_name)}</span>
    <span class="pill">Target: {_html.escape(target)}</span>
    <span class="pill">Typ: {_html.escape(typ)}</span>
  </div>

  <section class="card">
    <h2>Metryki</h2>
    {metrics_html}
  </section>

  <section class="card">
    <h2>Ważność cech</h2>
    {img_html}
    <div style="margin-top:12px">{features_table}</div>
  </section>

  <section class="card">
    <h2>Rekomendacje</h2>
    {rekomendacje_html}
  </section>

  <footer>
    Wygenerowano przez: {_html.escape(autor)}
  </footer>
</body>
</html>
    """.strip()

    out = Path(output_path)
    out.write_text(html_doc, encoding="utf-8")
    return str(out)

