"""Rutas de Variables Cualitativas."""
from flask import Blueprint, request, jsonify
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.gestor_datos import obtener_limpio, hay_datos, columnas_categoricas
from modules.utilidades_graficas import fig_a_json, ESQUEMA_OSCURO, PALETA

rutas_cualitativas = Blueprint('cualitativas', __name__)


@rutas_cualitativas.route('/resumen', methods=['GET'])
def resumen():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df       = obtener_limpio()
    cols_cat = columnas_categoricas(df)
    resultado = []
    for col in cols_cat:
        vc   = df[col].astype(str).replace({'nan': '', '<NA>': '', 'None': ''}).value_counts()
        moda = str(vc.index[0]) if len(vc) else 'N/A'
        top  = [{'categoria': str(k), 'conteo': int(v), 'pct': round(v / max(len(df), 1) * 100, 2)}
                for k, v in vc.head(10).items()]
        resultado.append({
            'columna':    col,
            'unicos':     int(df[col].nunique()),
            'moda':       moda,
            'moda_conteo': int(vc.iloc[0]) if len(vc) else 0,
            'moda_pct':   round(vc.iloc[0] / max(len(df), 1) * 100, 2) if len(vc) else 0,
            'faltantes':  int(df[col].isnull().sum()),
            'top_categorias': top,
            'interpretacion': (
                f"La categoría más frecuente es <b>{moda}</b> con "
                f"{int(vc.iloc[0]) if len(vc) else 0} registros "
                f"({round(vc.iloc[0] / max(len(df), 1) * 100, 1) if len(vc) else 0}%)"
            ),
        })
    return jsonify({'columnas': cols_cat, 'datos': resultado})


@rutas_cualitativas.route('/barras', methods=['POST'])
def barras():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df     = obtener_limpio()
    col    = request.json.get('columna')
    top_n  = int(request.json.get('top_n', 15))
    if not col or col not in df.columns:
        return jsonify({'error': 'Columna inválida'}), 400
    vc  = df[col].astype(str).value_counts().head(top_n)
    fig = go.Figure(go.Bar(
        x=vc.index.tolist(), y=vc.values.tolist(),
        marker=dict(color=PALETA[:len(vc)], line=dict(color='rgba(0,0,0,0.3)', width=1)),
        text=vc.values.tolist(), textposition='outside', textfont=dict(color='#94a3b8'),
    ))
    fig.update_layout(**ESQUEMA_OSCURO, title=f'Frecuencia — {col}',
                      xaxis_title=col, yaxis_title='Frecuencia', height=430, bargap=0.18)
    return jsonify(fig_a_json(fig))


@rutas_cualitativas.route('/circular', methods=['POST'])
def circular():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df    = obtener_limpio()
    col   = request.json.get('columna')
    tipo  = request.json.get('tipo', 'circular')
    top_n = int(request.json.get('top_n', 10))
    if not col or col not in df.columns:
        return jsonify({'error': 'Columna inválida'}), 400
    vc   = df[col].astype(str).value_counts().head(top_n)
    hole = 0.42 if tipo == 'donut' else 0
    fig  = go.Figure(go.Pie(
        labels=vc.index.tolist(), values=vc.values.tolist(), hole=hole,
        marker=dict(colors=PALETA[:len(vc)], line=dict(color='#0f172a', width=2)),
        textinfo='label+percent', textfont=dict(size=11),
    ))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'), title_font=dict(color='#e2e8f0'),
        title=col, height=430, showlegend=True,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return jsonify(fig_a_json(fig))


@rutas_cualitativas.route('/mapa_arbol', methods=['POST'])
def mapa_arbol():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df    = obtener_limpio()
    col   = request.json.get('columna')
    top_n = int(request.json.get('top_n', 20))
    if not col or col not in df.columns:
        return jsonify({'error': 'Columna inválida'}), 400
    vc          = df[col].astype(str).value_counts().head(top_n).reset_index()
    vc.columns  = ['categoria', 'conteo']
    fig = px.treemap(vc, path=['categoria'], values='conteo',
                     color='conteo', color_continuous_scale=['#1e3a5f', '#38bdf8'])
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        title=f'Mapa de árbol — {col}', height=430,
        font=dict(color='#e2e8f0'), margin=dict(l=10, r=10, t=55, b=10),
    )
    return jsonify(fig_a_json(fig))


@rutas_cualitativas.route('/tabla_frecuencias', methods=['POST'])
def tabla_frecuencias():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df  = obtener_limpio()
    col = request.json.get('columna')
    if not col or col not in df.columns:
        return jsonify({'error': 'Columna inválida'}), 400
    vc      = df[col].astype(str).value_counts()
    total   = len(df)
    filas   = []
    acum    = 0
    for cat, conteo in vc.items():
        pct  = round(conteo / max(total, 1) * 100, 2)
        acum = round(acum + pct, 2)
        filas.append({
            'categoria':     str(cat),
            'frecuencia':    int(conteo),
            'pct_relativa':  pct,
            'pct_acumulada': min(acum, 100.0),
        })
    return jsonify({'filas': filas, 'total': total, 'columna': col})
