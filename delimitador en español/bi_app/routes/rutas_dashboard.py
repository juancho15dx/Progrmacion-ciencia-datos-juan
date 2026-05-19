"""Rutas del Dashboard Ejecutivo."""
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.gestor_datos import obtener_limpio, hay_datos, columnas_numericas, columnas_categoricas
from modules.utilidades_graficas import fig_a_json, ESQUEMA_OSCURO, PALETA

rutas_dashboard = Blueprint('dashboard', __name__)

ESQUEMA_MINI = {k: v for k, v in ESQUEMA_OSCURO.items()}
ESQUEMA_MINI['margin'] = dict(l=45, r=15, t=45, b=45)


@rutas_dashboard.route('/indicadores', methods=['GET'])
def indicadores():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df       = obtener_limpio()
    cols_num = columnas_numericas(df)
    cols_cat = columnas_categoricas(df)
    lista_kpi = []
    for col in cols_num[:6]:
        s = df[col].dropna()
        lista_kpi.append({
            'nombre': col,
            'suma':   round(float(s.sum()), 2),
            'media':  round(float(s.mean()), 2),
            'minimo': round(float(s.min()), 2),
            'maximo': round(float(s.max()), 2),
        })
    return jsonify({
        'indicadores':    lista_kpi,
        'total_filas':    len(df),
        'cant_numericas': len(cols_num),
        'cant_categoricas': len(cols_cat),
    })


@rutas_dashboard.route('/graficas', methods=['POST'])
def graficas():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df      = obtener_limpio().copy()
    filtros = (request.json or {}).get('filtros', {})
    for col, val in filtros.items():
        if col in df.columns and val and val != 'Todos':
            df = df[df[col].astype(str) == str(val)]

    cols_num = columnas_numericas(df)
    cols_cat = columnas_categoricas(df)
    resultados = {}

    if cols_num:
        col  = cols_num[0]
        vals = df[col].dropna().tolist()
        fig  = go.Figure(go.Scatter(
            y=vals, mode='lines+markers',
            line=dict(color='#38bdf8', width=2), marker=dict(size=3, color='#38bdf8'),
            fill='tozeroy', fillcolor='rgba(56,189,248,0.08)',
        ))
        fig.update_layout(**ESQUEMA_MINI, title=f'Tendencia — {col}', height=290)
        resultados['tendencia'] = fig_a_json(fig)

    if cols_cat:
        col = cols_cat[0]
        vc  = df[col].astype(str).value_counts().head(8)
        fig2 = go.Figure(go.Bar(
            x=vc.values.tolist(), y=vc.index.tolist(),
            orientation='h', marker=dict(color=PALETA[:len(vc)]),
        ))
        fig2.update_layout(**ESQUEMA_MINI, title=f'Top categorías — {col}', height=290)
        resultados['categorias'] = fig_a_json(fig2)

    if len(cols_num) >= 2:
        fig3 = go.Figure()
        for i, col in enumerate(cols_num[:4]):
            fig3.add_trace(go.Histogram(
                x=df[col].dropna().tolist(), name=col,
                marker_color=PALETA[i], opacity=0.75, nbinsx=22,
            ))
        fig3.update_layout(**ESQUEMA_MINI, barmode='overlay',
                           title='Distribuciones comparadas', height=290)
        resultados['distribuciones'] = fig_a_json(fig3)

    if cols_cat:
        col = cols_cat[0]
        vc  = df[col].astype(str).value_counts().head(6)
        fig4 = go.Figure(go.Pie(
            labels=vc.index.tolist(), values=vc.values.tolist(), hole=0.4,
            marker=dict(colors=PALETA[:len(vc)], line=dict(color='#0f172a', width=2)),
            textinfo='label+percent', textfont=dict(size=10),
        ))
        fig4.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            title=f'Proporción — {col}', height=290,
            font=dict(color='#94a3b8'), margin=dict(l=10, r=10, t=40, b=10),
        )
        resultados['proporcion'] = fig_a_json(fig4)

    return jsonify(resultados)


@rutas_dashboard.route('/opciones_filtro', methods=['GET'])
def opciones_filtro():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df       = obtener_limpio()
    cols_cat = columnas_categoricas(df)
    opciones = {}
    for col in cols_cat[:5]:
        vals         = df[col].dropna().astype(str).unique().tolist()[:30]
        opciones[col] = ['Todos'] + vals
    return jsonify({'opciones': opciones, 'cols_categoricas': cols_cat[:5]})
