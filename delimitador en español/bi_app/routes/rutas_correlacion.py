"""Rutas de Correlación."""
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.gestor_datos import obtener_limpio, hay_datos, columnas_numericas
from modules.utilidades_graficas import fig_a_json, ESQUEMA_OSCURO, PALETA

rutas_correlacion = Blueprint('correlacion', __name__)


@rutas_correlacion.route('/matriz', methods=['GET'])
def matriz():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df       = obtener_limpio()
    cols_num = columnas_numericas(df)
    if len(cols_num) < 2:
        return jsonify({'error': 'Se necesitan al menos 2 variables numéricas'}), 400

    corr   = df[cols_num].corr().round(3)
    z      = corr.values.tolist()
    etiq   = corr.columns.tolist()

    fig = go.Figure(go.Heatmap(
        z=z, x=etiq, y=etiq,
        colorscale=[[0,'#ef4444'],[0.25,'#7f1d1d'],[0.5,'#1e293b'],[0.75,'#164e63'],[1,'#38bdf8']],
        zmid=0, zmin=-1, zmax=1,
        text=[[round(v, 2) for v in fila] for fila in z],
        texttemplate='%{text}', textfont=dict(size=10, color='white'),
        hoverongaps=False, showscale=True,
    ))
    alto = max(400, len(cols_num) * 52 + 100)
    fig.update_layout(**ESQUEMA_OSCURO, title='Matriz de Correlación', height=alto)

    pares = []
    for i in range(len(etiq)):
        for j in range(i + 1, len(etiq)):
            v = corr.iloc[i, j]
            if not np.isnan(v):
                pares.append({
                    'var1':        etiq[i],
                    'var2':        etiq[j],
                    'correlacion': round(float(v), 4),
                    'absoluta':    round(abs(float(v)), 4),
                    'tipo':        'Positiva' if v > 0.1 else ('Negativa' if v < -0.1 else 'Independiente'),
                    'intensidad':  'Alta' if abs(v) > 0.7 else ('Moderada' if abs(v) > 0.4 else 'Baja'),
                })
    pares.sort(key=lambda x: x['absoluta'], reverse=True)

    interpretaciones = []
    for p in pares[:5]:
        pct = round(abs(p['correlacion']) * 100, 1)
        dir = 'positiva' if p['correlacion'] > 0 else 'negativa'
        interpretaciones.append(
            f"<b>{p['var1']}</b> tiene correlación {dir} {p['intensidad'].lower()} "
            f"con <b>{p['var2']}</b> ({pct}%)"
        )

    return jsonify({
        'mapa_calor':       fig_a_json(fig),
        'pares':            pares[:30],
        'interpretaciones': interpretaciones,
        'columnas':         etiq,
    })


@rutas_correlacion.route('/scatter_par', methods=['POST'])
def scatter_par():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df   = obtener_limpio()
    var1 = request.json.get('var1')
    var2 = request.json.get('var2')
    if not var1 or not var2 or var1 not in df.columns or var2 not in df.columns:
        return jsonify({'error': 'Variables inválidas'}), 400

    sub   = df[[var1, var2]].dropna()
    r_val = float(sub[var1].corr(sub[var2]))
    x, y  = sub[var1].tolist(), sub[var2].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                             marker=dict(color='#38bdf8', size=6, opacity=0.6,
                                         line=dict(color='#6366f1', width=0.5)), name='Datos'))
    if len(x) > 2:
        xn   = np.array(x); yn = np.array(y)
        coef = np.polyfit(xn, yn, 1)
        xr   = np.linspace(xn.min(), xn.max(), 100).tolist()
        yr   = np.poly1d(coef)(np.linspace(xn.min(), xn.max(), 100)).tolist()
        fig.add_trace(go.Scatter(x=xr, y=yr, mode='lines',
                                 line=dict(color='#f59e0b', width=2, dash='dash'), name='Tendencia'))
    fig.update_layout(**ESQUEMA_OSCURO, title=f'{var1} vs {var2}  |  r = {r_val:.3f}',
                      xaxis_title=var1, yaxis_title=var2, height=420)
    return jsonify({'figura': fig_a_json(fig), 'correlacion': round(r_val, 4)})
