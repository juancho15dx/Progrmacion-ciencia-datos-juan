from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.data_manager import get_clean, has_data, get_numeric_cols
from modules.chart_utils import fig_json, DARK_LAYOUT, PALETTE

stats_bp = Blueprint('stats', __name__)

@stats_bp.route('/summary', methods=['GET'])
def summary():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    num_cols = get_numeric_cols(df)
    if not num_cols: return jsonify({'error': 'No se encontraron variables numéricas'}), 400
    result = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0: continue
        result.append({
            'variable': col, 'count': int(len(s)),
            'mean': round(float(s.mean()),4), 'median': round(float(s.median()),4),
            'mode': round(float(s.mode().iloc[0]),4) if len(s.mode()) else None,
            'std': round(float(s.std()),4), 'variance': round(float(s.var()),4),
            'min': round(float(s.min()),4), 'max': round(float(s.max()),4),
            'range': round(float(s.max()-s.min()),4),
            'p25': round(float(s.quantile(0.25)),4), 'p75': round(float(s.quantile(0.75)),4),
            'p90': round(float(s.quantile(0.90)),4),
            'skewness': round(float(s.skew()),4), 'kurtosis': round(float(s.kurtosis()),4),
            'missing': int(df[col].isnull().sum()),
        })
    return jsonify({'stats': result, 'columns': num_cols})

@stats_bp.route('/boxplot', methods=['POST'])
def boxplot():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    cols = request.json.get('columns', get_numeric_cols(df)[:8])
    cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols: return jsonify({'error': 'No hay columnas numéricas seleccionadas'}), 400
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Box(y=df[col].dropna().tolist(), name=col,
                             marker_color=PALETTE[i%len(PALETTE)], boxmean='sd',
                             line_color=PALETTE[i%len(PALETTE)]))
    fig.update_layout(**DARK_LAYOUT, title='Boxplots — Variables Numéricas', height=440, showlegend=True)
    return jsonify(fig_json(fig))

@stats_bp.route('/histogram', methods=['POST'])
def histogram():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    col = request.json.get('column')
    if not col or col not in df.columns: return jsonify({'error': 'Columna inválida'}), 400
    s = df[col].dropna()
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Histograma', 'Densidad KDE'])
    fig.add_trace(go.Histogram(x=s.tolist(), name='Frec.', marker_color='#00d4ff', opacity=0.85, nbinsx=30), row=1, col=1)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(s)
        xr = np.linspace(float(s.min()), float(s.max()), 200)
        fig.add_trace(go.Scatter(x=xr.tolist(), y=kde(xr).tolist(), mode='lines',
                                 line=dict(color='#7c3aed', width=3), name='KDE'), row=1, col=2)
    except Exception:
        hist, edges = np.histogram(s, bins=30, density=True)
        centers = ((edges[:-1]+edges[1:])/2).tolist()
        fig.add_trace(go.Scatter(x=centers, y=hist.tolist(), mode='lines',
                                 line=dict(color='#7c3aed', width=3), name='Dens.'), row=1, col=2)
    fig.update_layout(**DARK_LAYOUT, title=f'Distribución: {col}', height=420, showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)')
    return jsonify(fig_json(fig))

@stats_bp.route('/violin', methods=['POST'])
def violin():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    cols = request.json.get('columns', get_numeric_cols(df)[:6])
    cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols: return jsonify({'error': 'Sin columnas válidas'}), 400
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Violin(y=df[col].dropna().tolist(), name=col,
                                box_visible=True, meanline_visible=True,
                                fillcolor=PALETTE[i%len(PALETTE)], opacity=0.75,
                                line_color=PALETTE[i%len(PALETTE)]))
    fig.update_layout(**DARK_LAYOUT, title='Violin Plots', height=440)
    return jsonify(fig_json(fig))

@stats_bp.route('/scatter', methods=['POST'])
def scatter():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    x_col = request.json.get('x'); y_col = request.json.get('y')
    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return jsonify({'error': 'Columnas inválidas'}), 400
    sub = df[[x_col, y_col]].dropna()
    fig = go.Figure(go.Scatter(x=sub[x_col].tolist(), y=sub[y_col].tolist(), mode='markers',
                               marker=dict(color='#00d4ff', size=6, opacity=0.65,
                                           line=dict(color='#7c3aed', width=0.5))))
    fig.update_layout(**DARK_LAYOUT, title=f'Dispersión: {x_col} vs {y_col}',
                      xaxis_title=x_col, yaxis_title=y_col, height=420)
    return jsonify(fig_json(fig))
