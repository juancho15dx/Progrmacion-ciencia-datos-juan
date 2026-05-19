from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.data_manager import get_clean, has_data, get_numeric_cols, get_categorical_cols
from modules.chart_utils import fig_json, DARK_LAYOUT, PALETTE

dash_bp = Blueprint('dashboard', __name__)

SMALL = {k:v for k,v in DARK_LAYOUT.items()}; SMALL["margin"] = dict(l=45,r=15,t=45,b=45)

@dash_bp.route('/kpis', methods=['GET'])
def kpis():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    num_cols = get_numeric_cols(df); cat_cols = get_categorical_cols(df)
    kpi_list = []
    for col in num_cols[:6]:
        s = df[col].dropna()
        kpi_list.append({'name': col, 'sum': round(float(s.sum()),2),
                         'mean': round(float(s.mean()),2),
                         'min': round(float(s.min()),2), 'max': round(float(s.max()),2)})
    return jsonify({'kpis': kpi_list, 'total_rows': len(df),
                    'num_count': len(num_cols), 'cat_count': len(cat_cols)})

@dash_bp.route('/charts', methods=['POST'])
def charts():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean().copy()
    filters = (request.json or {}).get('filters', {})
    for col, val in filters.items():
        if col in df.columns and val and val != 'Todos':
            df = df[df[col].astype(str) == str(val)]

    num_cols = get_numeric_cols(df); cat_cols = get_categorical_cols(df)
    results = {}

    if num_cols:
        col = num_cols[0]; vals = df[col].dropna().tolist()
        fig = go.Figure(go.Scatter(y=vals, mode='lines+markers',
                                   line=dict(color='#00d4ff',width=2), marker=dict(size=3,color='#00d4ff'),
                                   fill='tozeroy', fillcolor='rgba(0,212,255,0.08)'))
        fig.update_layout(**SMALL, title=f'Tendencia — {col}', height=290)
        results['trend'] = fig_json(fig)

    if cat_cols:
        col = cat_cols[0]; vc = df[col].astype(str).value_counts().head(8)
        fig2 = go.Figure(go.Bar(x=vc.values.tolist(), y=vc.index.tolist(),
                                orientation='h', marker=dict(color=PALETTE[:len(vc)])))
        fig2.update_layout(**SMALL, title=f'Top categorías — {col}', height=290)
        results['categorical'] = fig_json(fig2)

    if len(num_cols) >= 2:
        fig3 = go.Figure()
        for i, col in enumerate(num_cols[:4]):
            fig3.add_trace(go.Histogram(x=df[col].dropna().tolist(), name=col,
                                        marker_color=PALETTE[i], opacity=0.75, nbinsx=22))
        fig3.update_layout(**SMALL, barmode='overlay', title='Distribuciones comparadas', height=290)
        results['distribution'] = fig_json(fig3)

    if cat_cols:
        col = cat_cols[0]; vc = df[col].astype(str).value_counts().head(6)
        fig4 = go.Figure(go.Pie(labels=vc.index.tolist(), values=vc.values.tolist(), hole=0.4,
                                marker=dict(colors=PALETTE[:len(vc)], line=dict(color='#0f172a',width=2)),
                                textinfo='label+percent', textfont=dict(size=10)))
        fig4.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                           title=f'Proporción — {col}', height=290,
                           font=dict(color='#94a3b8'), margin=dict(l=10,r=10,t=40,b=10))
        results['pie'] = fig_json(fig4)

    return jsonify(results)

@dash_bp.route('/filter_options', methods=['GET'])
def filter_options():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean(); cat_cols = get_categorical_cols(df)
    options = {}
    for col in cat_cols[:5]:
        vals = df[col].dropna().astype(str).unique().tolist()[:30]
        options[col] = ['Todos'] + vals
    return jsonify({'options': options, 'cat_cols': cat_cols[:5]})
