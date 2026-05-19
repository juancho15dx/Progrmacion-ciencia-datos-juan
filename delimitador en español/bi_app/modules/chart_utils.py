"""
Chart utilities - fixes Plotly template serialization bug
"""
import json
import pandas as pd
import numpy as np


def fig_json(fig):
    """
    Convert Plotly figure to JSON-safe dict for the frontend.
    IMPORTANT: replaces the serialized template object with the string name
    so Plotly.js renders it correctly.
    """
    d = json.loads(fig.to_json())
    # Replace huge template dict with string name so Plotly.js uses its built-in theme
    if isinstance(d.get('layout', {}).get('template'), dict):
        d['layout']['template'] = 'plotly_dark'
    return d


DARK_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,23,42,0.95)',
    font=dict(color='#94a3b8', size=12),
    title_font=dict(color='#e2e8f0', size=14),
    margin=dict(l=55, r=20, t=55, b=55),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', linecolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.08)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', linecolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.08)'),
)

PALETTE = ['#00d4ff', '#7c3aed', '#10b981', '#f59e0b', '#ef4444',
           '#8b5cf6', '#06b6d4', '#84cc16', '#f97316', '#ec4899']


def read_any_file(file_storage):
    """
    Read a Flask FileStorage object as DataFrame.
    Handles: CSV (comma/semicolon/tab), Excel (.xlsx/.xls), encodings.
    Returns (df, error_string)
    """
    filename = file_storage.filename.lower() if file_storage.filename else ''

    # ---- Excel ----
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        try:
            file_storage.seek(0)
            df = pd.read_excel(file_storage)
            return df, None
        except Exception as e:
            return None, f'Error leyendo Excel: {e}'

    # ---- CSV ----
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    separators = [',', ';', '\t', '|']

    for enc in encodings:
        for sep in separators:
            try:
                file_storage.seek(0)
                df = pd.read_csv(file_storage, encoding=enc, sep=sep, engine='python')
                if len(df.columns) <= 1 and sep != separators[-1]:
                    continue  # wrong separator, try next
                # Fix decimal comma: columns that look numeric but stored as string
                for col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            converted = pd.to_numeric(
                                df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
                            if converted.notna().sum() / max(len(df), 1) >= 0.8:
                                df[col] = converted
                        except Exception:
                            pass
                return df, None
            except Exception:
                continue

    return None, 'No se pudo leer el archivo. Verifica que sea CSV o Excel válido.'


def auto_convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Try to convert string columns that look like numbers."""
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                converted = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
                ratio = converted.notna().sum() / max(len(df), 1)
                if ratio >= 0.8:
                    df[col] = converted
            except Exception:
                pass
    return df
