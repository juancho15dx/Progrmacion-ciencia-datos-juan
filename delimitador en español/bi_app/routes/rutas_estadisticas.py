"""
Rutas de Estadísticas — Resumen descriptivo y gráficas.
"""
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.gestor_datos import obtener_limpio, hay_datos, columnas_numericas
from modules.utilidades_graficas import fig_a_json, ESQUEMA_OSCURO, PALETA

rutas_estadisticas = Blueprint('estadisticas', __name__)


@rutas_estadisticas.route('/resumen', methods=['GET'])
def resumen():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df       = obtener_limpio()
    cols_num = columnas_numericas(df)
    if not cols_num:
        return jsonify({'error': 'No se encontraron variables numéricas en el dataset'}), 400

    resultados = []
    for col in cols_num:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        resultados.append({
            'variable':  col,
            'conteo':    int(len(s)),
            'media':     round(float(s.mean()), 4),
            'mediana':   round(float(s.median()), 4),
            'moda':      round(float(s.mode().iloc[0]), 4) if len(s.mode()) else None,
            'desv_std':  round(float(s.std()), 4),
            'varianza':  round(float(s.var()), 4),
            'minimo':    round(float(s.min()), 4),
            'maximo':    round(float(s.max()), 4),
            'rango':     round(float(s.max() - s.min()), 4),
            'p25':       round(float(s.quantile(0.25)), 4),
            'p75':       round(float(s.quantile(0.75)), 4),
            'p90':       round(float(s.quantile(0.90)), 4),
            'asimetria': round(float(s.skew()), 4),
            'curtosis':  round(float(s.kurtosis()), 4),
            'faltantes': int(df[col].isnull().sum()),
        })
    return jsonify({'estadisticas': resultados, 'columnas': cols_num})


@rutas_estadisticas.route('/boxplot', methods=['POST'])
def boxplot():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df   = obtener_limpio()
    cols = request.json.get('columnas', columnas_numericas(df)[:8])
    cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return jsonify({'error': 'No hay columnas numéricas seleccionadas'}), 400
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Box(
            y=df[col].dropna().tolist(), name=col,
            marker_color=PALETA[i % len(PALETA)],
            line_color=PALETA[i % len(PALETA)],
            boxmean='sd',
        ))
    fig.update_layout(**ESQUEMA_OSCURO, title='Boxplots — Variables Numéricas', height=440, showlegend=True)
    return jsonify(fig_a_json(fig))


@rutas_estadisticas.route('/histograma', methods=['POST'])
def histograma():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df  = obtener_limpio()
    col = request.json.get('columna')
    if not col or col not in df.columns:
        return jsonify({'error': 'Columna inválida'}), 400
    s   = df[col].dropna()
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Histograma', 'Densidad KDE'])
    fig.add_trace(go.Histogram(x=s.tolist(), name='Frecuencia',
                               marker_color='#38bdf8', opacity=0.85, nbinsx=30), row=1, col=1)
    try:
        from scipy.stats import gaussian_kde
        kde  = gaussian_kde(s)
        xr   = np.linspace(float(s.min()), float(s.max()), 200)
        fig.add_trace(go.Scatter(x=xr.tolist(), y=kde(xr).tolist(), mode='lines',
                                 line=dict(color='#6366f1', width=3), name='KDE'), row=1, col=2)
    except Exception:
        hist, bordes = np.histogram(s, bins=30, density=True)
        centros = ((bordes[:-1] + bordes[1:]) / 2).tolist()
        fig.add_trace(go.Scatter(x=centros, y=hist.tolist(), mode='lines',
                                 line=dict(color='#6366f1', width=3), name='Densidad'), row=1, col=2)
    fig.update_layout(**ESQUEMA_OSCURO, title=f'Distribución: {col}', height=420, showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)')
    return jsonify(fig_a_json(fig))


@rutas_estadisticas.route('/violin', methods=['POST'])
def violin():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df   = obtener_limpio()
    cols = request.json.get('columnas', columnas_numericas(df)[:6])
    cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return jsonify({'error': 'Sin columnas numéricas válidas'}), 400
    fig  = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Violin(
            y=df[col].dropna().tolist(), name=col,
            box_visible=True, meanline_visible=True,
            fillcolor=PALETA[i % len(PALETA)],
            opacity=0.75,
            line_color=PALETA[i % len(PALETA)],
        ))
    fig.update_layout(**ESQUEMA_OSCURO, title='Violin Plots', height=440)
    return jsonify(fig_a_json(fig))


@rutas_estadisticas.route('/dispersion', methods=['POST'])
def dispersion():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df    = obtener_limpio()
    col_x = request.json.get('eje_x')
    col_y = request.json.get('eje_y')
    if not col_x or not col_y or col_x not in df.columns or col_y not in df.columns:
        return jsonify({'error': 'Columnas inválidas'}), 400
    sub = df[[col_x, col_y]].dropna()
    fig = go.Figure(go.Scatter(
        x=sub[col_x].tolist(), y=sub[col_y].tolist(),
        mode='markers',
        marker=dict(color='#38bdf8', size=6, opacity=0.65, line=dict(color='#6366f1', width=0.5)),
    ))
    fig.update_layout(**ESQUEMA_OSCURO, title=f'Dispersión: {col_x} vs {col_y}',
                      xaxis_title=col_x, yaxis_title=col_y, height=420)
    return jsonify(fig_a_json(fig))
