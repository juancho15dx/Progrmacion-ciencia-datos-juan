"""
Rutas de Machine Learning — Entrenamiento, predicción por archivo y predicción manual.
"""
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys, base64, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.gestor_datos import (obtener_limpio, hay_datos, columnas_numericas,
                                   columnas_categoricas, guardar_modelo, obtener_modelo)
from modules.utilidades_graficas import fig_a_json, ESQUEMA_OSCURO, PALETA, leer_archivo

rutas_ml = Blueprint('ml', __name__)


@rutas_ml.route('/columnas', methods=['GET'])
def columnas():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df = obtener_limpio()
    return jsonify({
        'todas_columnas': df.columns.tolist(),
        'numericas':      columnas_numericas(df),
        'categoricas':    columnas_categoricas(df),
    })


@rutas_ml.route('/entrenar', methods=['POST'])
def entrenar():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    datos       = request.json or {}
    objetivo    = datos.get('objetivo')
    predictoras = datos.get('predictoras', [])
    tipo_ml     = datos.get('tipo_ml', 'regresion')

    if not objetivo or not predictoras:
        return jsonify({'error': 'Selecciona la variable objetivo y las predictoras'}), 400

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                                  accuracy_score, confusion_matrix)

    df = obtener_limpio().copy().dropna(subset=[objetivo])
    if len(df) < 10:
        return jsonify({'error': 'Datos insuficientes (mínimo 10 filas sin nulos en la variable objetivo)'}), 400

    predictoras_validas = [f for f in predictoras if f in df.columns and f != objetivo]
    if not predictoras_validas:
        return jsonify({'error': 'No hay variables predictoras válidas'}), 400

    X = df[predictoras_validas].copy()
    y = df[objetivo].copy()

    # Codificar variables categóricas en X
    codificadores = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna('_NULO_'))
            codificadores[col] = le
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())

    codificador_objetivo = None
    clases               = None

    if tipo_ml == 'clasificacion':
        codificador_objetivo = LabelEncoder()
        y_enc   = codificador_objetivo.fit_transform(y.astype(str))
        clases  = codificador_objetivo.classes_.tolist()
    else:
        y_enc  = pd.to_numeric(y, errors='coerce')
        mascara = y_enc.notna()
        X      = X[mascara]
        y_enc  = y_enc[mascara]

    if len(X) < 10:
        return jsonify({'error': 'Datos insuficientes tras conversión numérica'}), 400

    X_arr  = X.values
    y_arr  = y_enc.values if hasattr(y_enc, 'values') else np.array(y_enc)
    X_ent, X_pru, y_ent, y_pru = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42)

    escalador = StandardScaler()
    X_ent_e   = escalador.fit_transform(X_ent)
    X_pru_e   = escalador.transform(X_pru)

    metricas = {}
    fig_pred = fig_conf = fig_imp = None
    importancia = []

    if tipo_ml == 'regresion':
        modelo  = LinearRegression()
        modelo.fit(X_ent_e, y_ent)
        y_pred  = modelo.predict(X_pru_e)
        metricas = {
            'rmse': round(float(np.sqrt(mean_squared_error(y_pru, y_pred))), 4),
            'mae':  round(float(mean_absolute_error(y_pru, y_pred)), 4),
            'r2':   round(float(r2_score(y_pru, y_pred)), 4),
        }
        importancia = sorted(
            [{'variable': f, 'importancia': round(float(c), 4)} for f, c in zip(predictoras_validas, modelo.coef_)],
            key=lambda x: abs(x['importancia']), reverse=True
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pru.tolist(), y=y_pred.tolist(), mode='markers',
                                 marker=dict(color='#38bdf8', size=6, opacity=0.65), name='Predicciones'))
        mn, mx = float(min(y_pru.min(), y_pred.min())), float(max(y_pru.max(), y_pred.max()))
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
                                 line=dict(color='#f59e0b', dash='dash', width=2), name='Ideal'))
        fig.update_layout(**ESQUEMA_OSCURO, title='Real vs Predicho',
                          xaxis_title='Valor real', yaxis_title='Valor predicho', height=380)
        fig_pred = fig_a_json(fig)
    else:
        modelo  = LogisticRegression(max_iter=1000, random_state=42)
        modelo.fit(X_ent_e, y_ent)
        y_pred  = modelo.predict(X_pru_e)
        metricas = {'exactitud': round(float(accuracy_score(y_pru, y_pred)), 4)}
        cm       = confusion_matrix(y_pru, y_pred)
        etiq_cls = [str(c) for c in (clases or sorted(set(y_pru.tolist())))]
        fig_cm   = go.Figure(go.Heatmap(
            z=cm.tolist(), x=etiq_cls, y=etiq_cls,
            colorscale=[[0, '#1e293b'], [1, '#38bdf8']],
            text=cm.tolist(), texttemplate='%{text}', textfont=dict(size=12, color='white'),
        ))
        fig_cm.update_layout(**ESQUEMA_OSCURO, title='Matriz de Confusión', height=380)
        fig_conf = fig_a_json(fig_cm)
        if hasattr(modelo, 'coef_'):
            coef = np.abs(modelo.coef_).mean(axis=0) if modelo.coef_.ndim > 1 else np.abs(modelo.coef_[0])
            importancia = sorted(
                [{'variable': f, 'importancia': round(float(c), 4)} for f, c in zip(predictoras_validas, coef)],
                key=lambda x: abs(x['importancia']), reverse=True
            )

    if importancia:
        fig_i = go.Figure(go.Bar(
            x=[i['importancia'] for i in importancia],
            y=[i['variable']    for i in importancia],
            orientation='h',
            marker=dict(color=PALETA[:len(importancia)]),
            text=[str(i['importancia']) for i in importancia], textposition='outside',
        ))
        fig_i.update_layout(**ESQUEMA_OSCURO, title='Importancia de Variables',
                             height=max(320, len(importancia) * 36 + 100))
        fig_imp = fig_a_json(fig_i)

    bundle = {
        'modelo':               modelo,
        'escalador':            escalador,
        'predictoras':          predictoras_validas,
        'objetivo':             objetivo,
        'tipo_ml':              tipo_ml,
        'codificador_objetivo': codificador_objetivo,
        'codificadores':        codificadores,
    }
    guardar_modelo(bundle)
    modelo_b64 = base64.b64encode(pickle.dumps(bundle)).decode('utf-8')

    return jsonify({
        'exito':           True,
        'metricas':        metricas,
        'importancia':     importancia[:15],
        'fig_pred':        fig_pred,
        'fig_conf':        fig_conf,
        'fig_imp':         fig_imp,
        'modelo_b64':      modelo_b64,
        'muestras_ent':    len(X_ent),
        'muestras_pru':    len(X_pru),
        'clases':          clases,
        'predictoras_usadas': predictoras_validas,
        'tipo_ml':         tipo_ml,
    })


@rutas_ml.route('/predecir_archivo', methods=['POST'])
def predecir_archivo():
    """Predicción a partir de un archivo CSV o Excel."""
    bundle = obtener_modelo()
    if bundle is None:
        return jsonify({'error': 'No hay modelo entrenado. Entrena primero un modelo.'}), 400
    if 'archivo' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400

    archivo = request.files['archivo']
    df_pred, error = leer_archivo(archivo)
    if error:
        return jsonify({'error': error}), 400

    try:
        modelo         = bundle['modelo']
        escalador      = bundle['escalador']
        predictoras    = bundle['predictoras']
        tipo_ml        = bundle['tipo_ml']
        cod_obj        = bundle.get('codificador_objetivo')
        codificadores  = bundle.get('codificadores', {})

        faltantes = [f for f in predictoras if f not in df_pred.columns]
        if faltantes:
            return jsonify({'error': f'Columnas faltantes en el archivo: {", ".join(faltantes)}'}), 400

        X = df_pred[predictoras].copy()
        for col in X.columns:
            if col in codificadores:
                X[col] = codificadores[col].transform(X[col].astype(str).fillna('_NULO_'))
            elif not pd.api.types.is_numeric_dtype(X[col]):
                from sklearn.preprocessing import LabelEncoder
                le2     = LabelEncoder()
                X[col]  = le2.fit_transform(X[col].astype(str).fillna('_NULO_'))
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        X_e          = escalador.transform(X.values)
        predicciones = modelo.predict(X_e)
        etiq_pred    = cod_obj.inverse_transform(predicciones) if (tipo_ml == 'clasificacion' and cod_obj) else predicciones

        df_resultado = df_pred.copy()
        df_resultado['PREDICCION'] = etiq_pred

        if tipo_ml == 'clasificacion':
            from collections import Counter
            cnt = Counter(etiq_pred.tolist() if hasattr(etiq_pred, 'tolist') else list(etiq_pred))
            fig = go.Figure(go.Bar(x=list(cnt.keys()), y=list(cnt.values()), marker=dict(color=PALETA)))
        else:
            lista_pred = etiq_pred.tolist() if hasattr(etiq_pred, 'tolist') else list(etiq_pred)
            fig = go.Figure(go.Histogram(x=lista_pred, marker_color='#6366f1', nbinsx=30))
        fig.update_layout(**ESQUEMA_OSCURO, title='Distribución de Predicciones', height=350)

        columnas_res  = df_resultado.columns.tolist()
        datos_res     = [
            ['' if pd.isna(v) else str(v) for v in fila]
            for _, fila in df_resultado.head(100).iterrows()
        ]
        return jsonify({
            'exito':              True,
            'total_predicciones': len(etiq_pred),
            'vista_previa':       {'columnas': columnas_res, 'datos': datos_res, 'total': len(df_resultado)},
            'figura':             fig_a_json(fig),
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'detalle': traceback.format_exc()}), 500


@rutas_ml.route('/predecir_manual', methods=['POST'])
def predecir_manual():
    """Predicción a partir de valores ingresados manualmente."""
    bundle = obtener_modelo()
    if bundle is None:
        return jsonify({'error': 'No hay modelo entrenado. Entrena primero un modelo.'}), 400

    datos_entrada = request.json or {}
    valores_manual = datos_entrada.get('valores', {})

    try:
        modelo        = bundle['modelo']
        escalador     = bundle['escalador']
        predictoras   = bundle['predictoras']
        tipo_ml       = bundle['tipo_ml']
        cod_obj       = bundle.get('codificador_objetivo')
        codificadores = bundle.get('codificadores', {})

        faltantes = [f for f in predictoras if f not in valores_manual]
        if faltantes:
            return jsonify({'error': f'Valores faltantes para: {", ".join(faltantes)}'}), 400

        fila = {}
        for col in predictoras:
            val = valores_manual[col]
            if col in codificadores:
                try:
                    fila[col] = codificadores[col].transform([str(val)])[0]
                except Exception:
                    # Valor desconocido: usar 0
                    fila[col] = 0
            else:
                try:
                    fila[col] = float(val)
                except Exception:
                    fila[col] = 0.0

        X_fila = np.array([[fila[c] for c in predictoras]])
        X_esc  = escalador.transform(X_fila)
        pred   = modelo.predict(X_esc)

        if tipo_ml == 'clasificacion' and cod_obj:
            resultado = str(cod_obj.inverse_transform(pred)[0])
            # Probabilidades si el modelo las soporta
            proba = None
            if hasattr(modelo, 'predict_proba'):
                proba_raw = modelo.predict_proba(X_esc)[0]
                proba = {str(cls): round(float(p) * 100, 2)
                         for cls, p in zip(cod_obj.classes_, proba_raw)}
            return jsonify({
                'exito':        True,
                'prediccion':   resultado,
                'probabilidades': proba,
                'tipo_ml':      tipo_ml,
            })
        else:
            resultado = round(float(pred[0]), 4)
            return jsonify({
                'exito':      True,
                'prediccion': resultado,
                'tipo_ml':    tipo_ml,
            })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'detalle': traceback.format_exc()}), 500
