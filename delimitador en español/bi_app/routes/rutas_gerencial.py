"""
Rutas Gerenciales — Módulos avanzados de BI para toma de decisiones:
  1. Análisis de Pareto (regla 80/20)
  2. Detección de anomalías / valores atípicos
  3. Series de tiempo y tendencias
  4. Análisis de segmentación (clustering K-Means)
  5. Tabla dinámica gerencial
  6. Indicadores de rendimiento KPI avanzados
"""
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.gestor_datos import obtener_limpio, hay_datos, columnas_numericas, columnas_categoricas
from modules.utilidades_graficas import fig_a_json, ESQUEMA_OSCURO, PALETA

rutas_gerencial = Blueprint('gerencial', __name__)

ESQ = ESQUEMA_OSCURO


# ─────────────────────────────────────────────
# 1. ANÁLISIS DE PARETO
# ─────────────────────────────────────────────
@rutas_gerencial.route('/pareto', methods=['POST'])
def pareto():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df   = obtener_limpio()
    col_val  = request.json.get('columna_valor')
    col_cat  = request.json.get('columna_categoria')
    if not col_val or not col_cat:
        return jsonify({'error': 'Selecciona columna de valor y categoría'}), 400

    agrupado = df.groupby(col_cat)[col_val].sum().sort_values(ascending=False).head(20)
    total    = agrupado.sum()
    acum_pct = (agrupado.cumsum() / total * 100).round(2)
    pct_ind  = (agrupado / total * 100).round(2)

    # Índice del 80%
    corte_80 = int((acum_pct <= 80).sum())

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colores = [PALETA[0] if i <= corte_80 else PALETA[4] for i in range(len(agrupado))]
    fig.add_trace(go.Bar(x=agrupado.index.tolist(), y=agrupado.values.tolist(),
                         name='Valor', marker=dict(color=colores),
                         text=[f'{v:,.0f}' for v in agrupado.values],
                         textposition='outside'), secondary_y=False)
    fig.add_trace(go.Scatter(x=agrupado.index.tolist(), y=acum_pct.tolist(),
                              mode='lines+markers', name='Acumulado %',
                              line=dict(color='#f59e0b', width=2),
                              marker=dict(size=6)), secondary_y=True)
    fig.add_hline(y=80, line_dash='dash', line_color='#ef4444',
                  annotation_text='80%', secondary_y=True)
    fig.update_layout(**ESQ, title=f'Análisis de Pareto — {col_cat} por {col_val}', height=460)
    fig.update_yaxes(title_text='Valor acumulado', secondary_y=False)
    fig.update_yaxes(title_text='Porcentaje acumulado (%)', secondary_y=True, range=[0, 105])

    categorias_vitales = agrupado.index[:corte_80+1].tolist()
    pct_vital          = round(float(acum_pct.iloc[corte_80]), 1)

    return jsonify({
        'figura':             fig_a_json(fig),
        'categorias_vitales': categorias_vitales,
        'pct_valor_vital':    pct_vital,
        'total_categorias':   len(agrupado),
        'categorias_vitales_n': corte_80 + 1,
        'tabla': [
            {'categoria': str(k), 'valor': round(float(v),2),
             'pct_individual': round(float(pct_ind[k]),2),
             'pct_acumulado':  round(float(acum_pct[k]),2)}
            for k, v in agrupado.items()
        ],
    })


# ─────────────────────────────────────────────
# 2. DETECCIÓN DE ANOMALÍAS
# ─────────────────────────────────────────────
@rutas_gerencial.route('/anomalias', methods=['POST'])
def anomalias():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df      = obtener_limpio()
    columna = request.json.get('columna')
    metodo  = request.json.get('metodo', 'iqr')   # 'iqr' o 'zscore'
    if not columna or columna not in df.columns:
        return jsonify({'error': 'Columna inválida'}), 400

    s = df[columna].dropna()
    idx_orig = s.index

    if metodo == 'iqr':
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR    = Q3 - Q1
        lim_inf, lim_sup = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        es_anomalia = (s < lim_inf) | (s > lim_sup)
    else:  # zscore
        z_scores    = np.abs((s - s.mean()) / s.std())
        es_anomalia = z_scores > 3
        lim_inf     = float(s.mean() - 3*s.std())
        lim_sup     = float(s.mean() + 3*s.std())

    normales    = s[~es_anomalia]
    atipicos    = s[es_anomalia]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(normales))), y=normales.tolist(),
                              mode='markers', name='Normal',
                              marker=dict(color=PALETA[0], size=5, opacity=0.6)))
    if len(atipicos):
        posiciones_at = [list(s.index).index(i) for i in atipicos.index if i in list(s.index)]
        fig.add_trace(go.Scatter(x=posiciones_at, y=atipicos.tolist(),
                                  mode='markers', name='Anomalía',
                                  marker=dict(color=PALETA[4], size=9, symbol='x',
                                               line=dict(width=2, color=PALETA[4]))))
    fig.add_hline(y=lim_sup, line_dash='dash', line_color='#ef4444', annotation_text=f'Límite sup: {lim_sup:.2f}')
    fig.add_hline(y=lim_inf, line_dash='dash', line_color='#ef4444', annotation_text=f'Límite inf: {lim_inf:.2f}')
    fig.update_layout(**ESQ, title=f'Detección de Anomalías — {columna} ({metodo.upper()})', height=420)

    return jsonify({
        'figura':          fig_a_json(fig),
        'total_registros': int(len(s)),
        'total_anomalias': int(es_anomalia.sum()),
        'pct_anomalias':   round(float(es_anomalia.mean()*100), 2),
        'limite_inferior': round(float(lim_inf), 4),
        'limite_superior': round(float(lim_sup), 4),
        'valores_atipicos': [round(float(v),4) for v in atipicos.tolist()[:20]],
    })


# ─────────────────────────────────────────────
# 3. SERIES DE TIEMPO Y TENDENCIAS
# ─────────────────────────────────────────────
@rutas_gerencial.route('/tendencia', methods=['POST'])
def tendencia():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df      = obtener_limpio()
    col_val = request.json.get('columna_valor')
    col_per = request.json.get('columna_periodo')  # puede ser None
    ventana = int(request.json.get('ventana_media', 7))
    if not col_val or col_val not in df.columns:
        return jsonify({'error': 'Columna de valor inválida'}), 400

    if col_per and col_per in df.columns:
        try:
            df2 = df[[col_per, col_val]].copy()
            df2[col_per] = pd.to_datetime(df2[col_per], errors='coerce')
            df2 = df2.dropna().sort_values(col_per)
            eje_x = df2[col_per].astype(str).tolist()
            vals  = pd.to_numeric(df2[col_val], errors='coerce').dropna().tolist()
        except Exception:
            eje_x = list(range(len(df)))
            vals  = pd.to_numeric(df[col_val], errors='coerce').dropna().tolist()
    else:
        vals  = pd.to_numeric(df[col_val], errors='coerce').dropna().tolist()
        eje_x = list(range(len(vals)))

    serie = pd.Series(vals)
    media_movil = serie.rolling(window=min(ventana, len(serie))).mean().tolist()

    # Tendencia lineal
    x_num = np.arange(len(vals))
    coef  = np.polyfit(x_num, vals, 1)
    tend  = np.poly1d(coef)(x_num).tolist()
    cambio_pct = round((vals[-1] - vals[0]) / abs(vals[0]) * 100, 2) if vals[0] != 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eje_x, y=vals, mode='lines',
                              name='Valor real', line=dict(color=PALETA[0], width=1.5),
                              fill='tozeroy', fillcolor=f'rgba(56,189,248,0.08)'))
    fig.add_trace(go.Scatter(x=eje_x, y=media_movil, mode='lines',
                              name=f'Media móvil ({ventana})', line=dict(color=PALETA[2], width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=eje_x, y=tend, mode='lines',
                              name='Tendencia lineal', line=dict(color=PALETA[3], width=2, dash='dash')))
    fig.update_layout(**ESQ, title=f'Análisis de Tendencia — {col_val}', height=440)

    return jsonify({
        'figura':       fig_a_json(fig),
        'valor_inicial': round(float(vals[0]), 4) if vals else None,
        'valor_final':   round(float(vals[-1]), 4) if vals else None,
        'cambio_pct':    cambio_pct,
        'media_total':   round(float(serie.mean()), 4),
        'maximo':        round(float(serie.max()), 4),
        'minimo':        round(float(serie.min()), 4),
        'pendiente':     round(float(coef[0]), 6),
        'direccion':     'Creciente' if coef[0] > 0 else 'Decreciente',
    })


# ─────────────────────────────────────────────
# 4. SEGMENTACIÓN / CLUSTERING
# ─────────────────────────────────────────────
@rutas_gerencial.route('/segmentacion', methods=['POST'])
def segmentacion():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df       = obtener_limpio()
    col_x    = request.json.get('col_x')
    col_y    = request.json.get('col_y')
    n_grupos = int(request.json.get('n_grupos', 3))
    if not col_x or not col_y or col_x not in df.columns or col_y not in df.columns:
        return jsonify({'error': 'Selecciona dos columnas numéricas'}), 400

    sub = df[[col_x, col_y]].dropna()
    if len(sub) < n_grupos * 2:
        return jsonify({'error': 'Datos insuficientes para segmentar'}), 400

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        escalador = StandardScaler()
        X_esc     = escalador.fit_transform(sub.values)
        km        = KMeans(n_clusters=n_grupos, random_state=42, n_init=10)
        etiquetas = km.fit_predict(X_esc)
    except Exception as e:
        return jsonify({'error': f'Error en clustering: {e}'}), 500

    colores_seg = PALETA[:n_grupos]
    fig = go.Figure()
    for g in range(n_grupos):
        mask = etiquetas == g
        fig.add_trace(go.Scatter(
            x=sub[col_x][mask].tolist(), y=sub[col_y][mask].tolist(),
            mode='markers', name=f'Segmento {g+1}',
            marker=dict(color=colores_seg[g], size=7, opacity=0.75,
                        line=dict(width=0.5, color='rgba(0,0,0,0.3)'))))

    centroides_orig = escalador.inverse_transform(km.cluster_centers_)
    fig.add_trace(go.Scatter(
        x=centroides_orig[:, 0].tolist(), y=centroides_orig[:, 1].tolist(),
        mode='markers', name='Centroides',
        marker=dict(symbol='star', size=16, color='white',
                    line=dict(width=2, color='#f59e0b'))))
    fig.update_layout(**ESQ, title=f'Segmentación — {col_x} vs {col_y} ({n_grupos} grupos)',
                      xaxis_title=col_x, yaxis_title=col_y, height=460)

    resumen_grupos = []
    sub2 = sub.copy(); sub2['grupo'] = etiquetas
    for g in range(n_grupos):
        sg = sub2[sub2['grupo'] == g]
        resumen_grupos.append({
            'grupo':      g + 1,
            'registros':  int(len(sg)),
            'pct':        round(len(sg)/len(sub)*100, 1),
            f'media_{col_x}': round(float(sg[col_x].mean()), 4),
            f'media_{col_y}': round(float(sg[col_y].mean()), 4),
        })

    return jsonify({'figura': fig_a_json(fig), 'resumen_grupos': resumen_grupos,
                    'total_registros': len(sub)})


# ─────────────────────────────────────────────
# 5. TABLA DINÁMICA GERENCIAL
# ─────────────────────────────────────────────
@rutas_gerencial.route('/tabla_dinamica', methods=['POST'])
def tabla_dinamica():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df       = obtener_limpio()
    filas    = request.json.get('filas')
    columnas = request.json.get('columnas')
    valores  = request.json.get('valores')
    agregacion = request.json.get('agregacion', 'suma')

    if not filas or not valores:
        return jsonify({'error': 'Selecciona filas y columna de valores'}), 400

    func_map = {'suma': 'sum', 'media': 'mean', 'conteo': 'count',
                'maximo': 'max', 'minimo': 'min', 'mediana': 'median'}
    func     = func_map.get(agregacion, 'sum')

    try:
        pivot = pd.pivot_table(
            df, values=valores,
            index=[filas],
            columns=[columnas] if columnas and columnas in df.columns else None,
            aggfunc=func, fill_value=0,
        ).round(2)
        pivot.columns = [str(c) for c in pivot.columns]
        encabezados   = [str(filas)] + pivot.columns.tolist()
        datos_tabla   = []
        for idx, row in pivot.iterrows():
            fila = [str(idx)] + [round(float(v), 2) for v in row.values]
            datos_tabla.append(fila)

        # Gráfica de barras agrupadas
        fig = go.Figure()
        for i, col in enumerate(pivot.columns[:10]):
            fig.add_trace(go.Bar(
                name=str(col), x=[str(i) for i in pivot.index.tolist()],
                y=pivot[col].tolist(), marker=dict(color=PALETA[i % len(PALETA)]),
            ))
        fig.update_layout(**ESQ, barmode='group',
                          title=f'Tabla Dinámica — {valores} por {filas}',
                          height=420, xaxis_title=filas, yaxis_title=valores)

        return jsonify({
            'figura':      fig_a_json(fig),
            'encabezados': encabezados,
            'datos':       datos_tabla,
            'total_filas': len(datos_tabla),
        })
    except Exception as e:
        return jsonify({'error': f'Error al generar tabla dinámica: {e}'}), 500


# ─────────────────────────────────────────────
# 6. KPIs AVANZADOS Y SEMÁFOROS
# ─────────────────────────────────────────────
@rutas_gerencial.route('/kpis_avanzados', methods=['GET'])
def kpis_avanzados():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df       = obtener_limpio()
    cols_num = columnas_numericas(df)
    if not cols_num:
        return jsonify({'error': 'No hay variables numéricas'}), 400

    kpis = []
    for col in cols_num[:8]:
        s     = df[col].dropna()
        media = float(s.mean())
        std   = float(s.std())
        ultimo = float(s.iloc[-1]) if len(s) > 0 else media
        cambio = round((s.iloc[-1] - s.iloc[0]) / abs(s.iloc[0]) * 100, 2) if len(s) > 1 and s.iloc[0] != 0 else 0
        cv     = round(std / abs(media) * 100, 2) if media != 0 else 0

        # Semáforo: verde si está dentro de 1 std, amarillo 1-2 std, rojo > 2 std
        distancia_std = abs(ultimo - media) / std if std > 0 else 0
        semaforo = 'verde' if distancia_std <= 1 else ('amarillo' if distancia_std <= 2 else 'rojo')

        kpis.append({
            'nombre':          col,
            'ultimo_valor':    round(ultimo, 4),
            'media':           round(media, 4),
            'desv_std':        round(std, 4),
            'minimo':          round(float(s.min()), 4),
            'maximo':          round(float(s.max()), 4),
            'cambio_pct':      cambio,
            'coef_variacion':  cv,
            'semaforo':        semaforo,
            'percentil_actual': round(float((s < ultimo).mean() * 100), 1),
        })

    # Gráfica gauge para las primeras 4 variables
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type":"indicator"},{"type":"indicator"}],[{"type":"indicator"},{"type":"indicator"}]],
        subplot_titles=[k['nombre'] for k in kpis[:4]],
    )
    posiciones = [(1,1),(1,2),(2,1),(2,2)]
    for idx, (k, pos) in enumerate(zip(kpis[:4], posiciones)):
        color_gauge = {'verde':'#10b981','amarillo':'#f59e0b','rojo':'#ef4444'}[k['semaforo']]
        fig.add_trace(go.Indicator(
            mode='gauge+number+delta',
            value=k['ultimo_valor'],
            delta={'reference': k['media'], 'relative': True, 'valueformat': '.1%'},
            gauge={
                'axis': {'range': [k['minimo'], k['maximo']]},
                'bar':  {'color': color_gauge},
                'steps': [
                    {'range': [k['minimo'], k['media'] - k['desv_std']], 'color': 'rgba(239,68,68,0.15)'},
                    {'range': [k['media'] - k['desv_std'], k['media'] + k['desv_std']], 'color': 'rgba(16,185,129,0.15)'},
                    {'range': [k['media'] + k['desv_std'], k['maximo']], 'color': 'rgba(239,68,68,0.15)'},
                ],
                'threshold': {'line': {'color': '#f59e0b','width': 3}, 'value': k['media']},
            },
            number={'font': {'color': color_gauge}},
        ), row=pos[0], col=pos[1])

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', size=11), height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        title_text='Velocímetros de KPI',
        title_font=dict(color='#e2e8f0', size=14),
    )

    return jsonify({'figura': fig_a_json(fig), 'kpis': kpis})


# ─────────────────────────────────────────────
# 7. ANÁLISIS COMPARATIVO ENTRE GRUPOS
# ─────────────────────────────────────────────
@rutas_gerencial.route('/comparativo_grupos', methods=['POST'])
def comparativo_grupos():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df       = obtener_limpio()
    col_grupo = request.json.get('columna_grupo')
    col_val   = request.json.get('columna_valor')
    if not col_grupo or not col_val or col_grupo not in df.columns or col_val not in df.columns:
        return jsonify({'error': 'Columnas inválidas'}), 400

    agrupado = df.groupby(col_grupo)[col_val].agg(['mean','median','std','min','max','count']).round(4)
    agrupado.columns = ['Media','Mediana','Desv.Std','Mínimo','Máximo','Conteo']

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Media por grupo', 'Distribución por grupo'])
    grupos  = agrupado.index.tolist()
    medias  = agrupado['Media'].tolist()
    fig.add_trace(go.Bar(x=grupos, y=medias, name='Media',
                          marker=dict(color=PALETA[:len(grupos)]),
                          text=[f'{v:,.2f}' for v in medias], textposition='outside'),
                  row=1, col=1)

    for i, grupo in enumerate(grupos[:8]):
        datos_g = df[df[col_grupo] == grupo][col_val].dropna()
        fig.add_trace(go.Box(y=datos_g.tolist(), name=str(grupo),
                              marker_color=PALETA[i % len(PALETA)], boxmean=True), row=1, col=2)

    fig.update_layout(**ESQ, title=f'Comparativo: {col_val} por {col_grupo}', height=440)

    tabla = agrupado.reset_index().rename(columns={col_grupo: 'Grupo'})
    return jsonify({
        'figura':  fig_a_json(fig),
        'tabla':   tabla.to_dict(orient='records'),
        'columnas': ['Grupo'] + list(agrupado.columns),
    })
