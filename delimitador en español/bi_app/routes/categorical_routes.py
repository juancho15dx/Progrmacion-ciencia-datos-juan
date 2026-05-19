from flask import Blueprint, request, jsonify
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.data_manager import get_clean, has_data, get_categorical_cols
from modules.chart_utils import fig_json, DARK_LAYOUT, PALETTE

cat_bp = Blueprint('categorical', __name__)

@cat_bp.route('/overview', methods=['GET'])
def overview():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    cat_cols = get_categorical_cols(df)
    result = []
    for col in cat_cols:
        vc = df[col].astype(str).replace({'nan':'','<NA>':'','None':''}).value_counts()
        mode = str(vc.index[0]) if len(vc) else 'N/A'
        top = [{'category': str(k), 'count': int(v), 'pct': round(v/max(len(df),1)*100,2)}
               for k,v in vc.head(10).items()]
        result.append({'column': col, 'unique': int(df[col].nunique()),
                       'mode': mode,
                       'mode_count': int(vc.iloc[0]) if len(vc) else 0,
                       'mode_pct': round(vc.iloc[0]/max(len(df),1)*100,2) if len(vc) else 0,
                       'missing': int(df[col].isnull().sum()),
                       'top_categories': top,
                       'insight': f"La categoría más frecuente es <b>{mode}</b> con {int(vc.iloc[0]) if len(vc) else 0} registros ({round(vc.iloc[0]/max(len(df),1)*100,1) if len(vc) else 0}%)"})
    return jsonify({'columns': cat_cols, 'data': result})

@cat_bp.route('/bar', methods=['POST'])
def bar_chart():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    col = request.json.get('column'); top_n = int(request.json.get('top_n',15))
    if not col or col not in df.columns: return jsonify({'error': 'Columna inválida'}), 400
    vc = df[col].astype(str).value_counts().head(top_n)
    fig = go.Figure(go.Bar(
        x=vc.index.tolist(), y=vc.values.tolist(),
        marker=dict(color=PALETTE[:len(vc)], line=dict(color='rgba(0,0,0,0.3)', width=1)),
        text=vc.values.tolist(), textposition='outside', textfont=dict(color='#94a3b8')))
    fig.update_layout(**DARK_LAYOUT, title=f'Frecuencia — {col}',
                      xaxis_title=col, yaxis_title='Frecuencia', height=430, bargap=0.18)
    return jsonify(fig_json(fig))

@cat_bp.route('/pie', methods=['POST'])
def pie_chart():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    col = request.json.get('column'); chart_type = request.json.get('type','pie'); top_n = int(request.json.get('top_n',10))
    if not col or col not in df.columns: return jsonify({'error': 'Columna inválida'}), 400
    vc = df[col].astype(str).value_counts().head(top_n)
    hole = 0.42 if chart_type == 'donut' else 0
    fig = go.Figure(go.Pie(labels=vc.index.tolist(), values=vc.values.tolist(), hole=hole,
                           marker=dict(colors=PALETTE[:len(vc)], line=dict(color='#0f172a', width=2)),
                           textinfo='label+percent', textfont=dict(size=11)))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#94a3b8'), title_font=dict(color='#e2e8f0'),
                      title=f'{col}', height=430, showlegend=True,
                      margin=dict(l=20,r=20,t=55,b=20))
    return jsonify(fig_json(fig))

@cat_bp.route('/treemap', methods=['POST'])
def treemap():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    col = request.json.get('column'); top_n = int(request.json.get('top_n',20))
    if not col or col not in df.columns: return jsonify({'error': 'Columna inválida'}), 400
    vc = df[col].astype(str).value_counts().head(top_n).reset_index()
    vc.columns = ['category','count']
    fig = px.treemap(vc, path=['category'], values='count',
                     color='count', color_continuous_scale=['#1e3a5f','#00d4ff'])
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      title=f'Treemap — {col}', height=430,
                      font=dict(color='#e2e8f0'), margin=dict(l=10,r=10,t=55,b=10))
    return jsonify(fig_json(fig))

@cat_bp.route('/frequency_table', methods=['POST'])
def frequency_table():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    col = request.json.get('column')
    if not col or col not in df.columns: return jsonify({'error': 'Columna inválida'}), 400
    vc = df[col].astype(str).value_counts()
    total = len(df); rows = []; cum = 0
    for cat, count in vc.items():
        pct = round(count/max(total,1)*100, 2); cum = round(cum+pct, 2)
        rows.append({'category': str(cat), 'frequency': int(count),
                     'relative_pct': pct, 'cumulative_pct': min(cum,100.0)})
    return jsonify({'rows': rows, 'total': total, 'column': col})
