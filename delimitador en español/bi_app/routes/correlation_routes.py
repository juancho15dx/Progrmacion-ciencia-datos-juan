from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.data_manager import get_clean, has_data, get_numeric_cols
from modules.chart_utils import fig_json, DARK_LAYOUT, PALETTE

corr_bp = Blueprint('correlation', __name__)

@corr_bp.route('/matrix', methods=['GET'])
def matrix():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    num_cols = get_numeric_cols(df)
    if len(num_cols) < 2: return jsonify({'error': 'Se necesitan al menos 2 variables numéricas'}), 400
    corr = df[num_cols].corr().round(3)
    z = corr.values.tolist()
    labs = corr.columns.tolist()

    fig = go.Figure(go.Heatmap(
        z=z, x=labs, y=labs,
        colorscale=[[0,'#ef4444'],[0.25,'#7f1d1d'],[0.5,'#1e293b'],[0.75,'#164e63'],[1,'#00d4ff']],
        zmid=0, zmin=-1, zmax=1,
        text=[[round(v,2) for v in row] for row in z],
        texttemplate='%{text}', textfont=dict(size=10, color='white'),
        hoverongaps=False, showscale=True,
    ))
    fig.update_layout(**DARK_LAYOUT, title='Matriz de Correlación',
                      height=max(400, len(num_cols)*52+100))

    pairs = []
    for i in range(len(labs)):
        for j in range(i+1, len(labs)):
            v = corr.iloc[i,j]
            if not np.isnan(v):
                pairs.append({
                    'var1': labs[i], 'var2': labs[j],
                    'correlation': round(float(v),4), 'abs': round(abs(float(v)),4),
                    'type': 'Positiva' if v>0.1 else ('Negativa' if v<-0.1 else 'Independiente'),
                    'strength': 'Alta' if abs(v)>0.7 else ('Moderada' if abs(v)>0.4 else 'Baja'),
                })
    pairs.sort(key=lambda x: x['abs'], reverse=True)

    insights = []
    for p in pairs[:5]:
        pct = round(abs(p['correlation'])*100, 1)
        d = 'positiva' if p['correlation']>0 else 'negativa'
        insights.append(f"📊 <b>{p['var1']}</b> tiene correlación {d} {p['strength'].lower()} con <b>{p['var2']}</b> ({pct}%)")

    return jsonify({'heatmap': fig_json(fig), 'pairs': pairs[:30], 'insights': insights, 'columns': labs})

@corr_bp.route('/scatter_pair', methods=['POST'])
def scatter_pair():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    var1 = request.json.get('var1'); var2 = request.json.get('var2')
    if not var1 or not var2 or var1 not in df.columns or var2 not in df.columns:
        return jsonify({'error': 'Variables inválidas'}), 400
    sub = df[[var1,var2]].dropna()
    r_val = float(sub[var1].corr(sub[var2]))
    x, y = sub[var1].tolist(), sub[var2].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                             marker=dict(color='#00d4ff', size=6, opacity=0.6,
                                         line=dict(color='#7c3aed',width=0.5)), name='Datos'))
    if len(x) > 2:
        xn = np.array(x); yn = np.array(y)
        z = np.polyfit(xn, yn, 1)
        xr = np.linspace(xn.min(), xn.max(), 100).tolist()
        yr = np.poly1d(z)(np.linspace(xn.min(), xn.max(), 100)).tolist()
        fig.add_trace(go.Scatter(x=xr, y=yr, mode='lines',
                                 line=dict(color='#f59e0b', width=2, dash='dash'), name=f'Tendencia'))
    fig.update_layout(**DARK_LAYOUT, title=f'{var1} vs {var2}  |  r = {r_val:.3f}',
                      xaxis_title=var1, yaxis_title=var2, height=420)
    return jsonify({'fig': fig_json(fig), 'correlation': round(r_val,4)})
