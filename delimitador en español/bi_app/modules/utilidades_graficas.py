"""
Utilidades Gráficas — Corrección del bug de template Plotly y lector universal de archivos.
"""
import json
import pandas as pd
import numpy as np


def fig_a_json(fig):
    """
    Convierte figura Plotly a dict JSON para el frontend.
    Reemplaza el objeto template serializado por el string del nombre,
    para que Plotly.js use su tema integrado correctamente.
    """
    d = json.loads(fig.to_json())
    if isinstance(d.get('layout', {}).get('template'), dict):
        d['layout']['template'] = 'plotly_dark'
    return d


ESQUEMA_OSCURO = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,23,42,0.95)',
    font=dict(color='#94a3b8', size=12, family='Inter,sans-serif'),
    title_font=dict(color='#e2e8f0', size=14, family='Space Grotesk,sans-serif'),
    margin=dict(l=55, r=20, t=55, b=55),
    xaxis=dict(
        gridcolor='rgba(255,255,255,0.06)',
        linecolor='rgba(255,255,255,0.1)',
        zerolinecolor='rgba(255,255,255,0.08)'
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.06)',
        linecolor='rgba(255,255,255,0.1)',
        zerolinecolor='rgba(255,255,255,0.08)'
    ),
)

PALETA = [
    '#38bdf8', '#6366f1', '#10b981', '#f59e0b',
    '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16',
    '#f97316', '#ec4899',
]


def leer_archivo(archivo_flask):
    """
    Lee un FileStorage de Flask y devuelve (DataFrame, error_str).
    Soporta: CSV con coma/punto y coma/tabulador, Excel .xlsx/.xls,
    múltiples encodings y decimales con coma.
    """
    nombre = archivo_flask.filename.lower() if archivo_flask.filename else ''

    # ── Excel ──────────────────────────────────────────────────────
    if nombre.endswith('.xlsx') or nombre.endswith('.xls'):
        try:
            archivo_flask.seek(0)
            df = pd.read_excel(archivo_flask)
            return df, None
        except Exception as e:
            return None, f'Error leyendo Excel: {e}'

    # ── CSV ────────────────────────────────────────────────────────
    codificaciones  = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    separadores     = [',', ';', '\t', '|']

    for cod in codificaciones:
        for sep in separadores:
            try:
                archivo_flask.seek(0)
                df = pd.read_csv(archivo_flask, encoding=cod, sep=sep, engine='python')
                if len(df.columns) <= 1 and sep != separadores[-1]:
                    continue  # separador incorrecto
                # Convertir decimales con coma
                for col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            convertido = pd.to_numeric(
                                df[col].astype(str).str.replace(',', '.', regex=False),
                                errors='coerce'
                            )
                            if convertido.notna().sum() / max(len(df), 1) >= 0.8:
                                df[col] = convertido
                        except Exception:
                            pass
                return df, None
            except Exception:
                continue

    return None, 'No se pudo leer el archivo. Verifica que sea CSV o Excel válido.'


def auto_convertir_numericos(df: pd.DataFrame) -> pd.DataFrame:
    """Intenta convertir columnas string que parecen números."""
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                convertido = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.', regex=False),
                    errors='coerce'
                )
                ratio = convertido.notna().sum() / max(len(df), 1)
                if ratio >= 0.8:
                    df[col] = convertido
            except Exception:
                pass
    return df
