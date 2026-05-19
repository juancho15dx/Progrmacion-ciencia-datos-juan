"""
Rutas de Datos — Carga, exploración y limpieza del dataset.
"""
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.gestor_datos import (
    guardar_dataset, obtener_original, obtener_limpio, guardar_limpio,
    hay_datos, df_a_dict_seguro, columnas_numericas, columnas_categoricas
)
from modules.utilidades_graficas import leer_archivo, auto_convertir_numericos

rutas_datos = Blueprint('datos', __name__)


@rutas_datos.route('/cargar', methods=['POST'])
def cargar():
    if 'archivo' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    archivo = request.files['archivo']
    nombre  = archivo.filename or ''
    ext     = nombre.lower().rsplit('.', 1)[-1] if '.' in nombre else ''
    if ext not in ('csv', 'xlsx', 'xls'):
        return jsonify({'error': 'Solo se admiten archivos CSV o Excel (.xlsx / .xls)'}), 400
    df, error = leer_archivo(archivo)
    if error:
        return jsonify({'error': error}), 400
    df = auto_convertir_numericos(df)
    guardar_dataset(df, nombre)
    return jsonify({
        'exito': True,
        'nombre_archivo': nombre,
        'filas': len(df),
        'columnas': len(df.columns),
    })


@rutas_datos.route('/resumen', methods=['GET'])
def resumen():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df   = obtener_limpio()
    orig = obtener_original()
    cols_num = columnas_numericas(df)
    cols_cat = columnas_categoricas(df)

    faltantes      = df.isnull().sum()
    pct_faltantes  = (faltantes / max(len(df), 1) * 100).round(2)
    duplicados     = int(df.duplicated().sum())
    total_celdas   = df.shape[0] * df.shape[1]
    total_faltantes = int(faltantes.sum())
    calidad_datos  = round((1 - total_faltantes / max(total_celdas, 1)) * 100, 1)

    ranking_faltantes = sorted(
        [{'columna': c, 'faltantes': int(faltantes[c]), 'porcentaje': float(pct_faltantes[c])}
         for c in df.columns if faltantes[c] > 0],
        key=lambda x: x['faltantes'], reverse=True
    )

    info_columnas = [
        {
            'nombre':       c,
            'tipo_dato':    str(df[c].dtype),
            'unicos':       int(df[c].nunique()),
            'faltantes':    int(faltantes[c]),
            'pct_faltante': float(pct_faltantes[c]),
            'tipo':         'Numérica' if c in cols_num else 'Categórica',
        }
        for c in df.columns
    ]

    mem = int(df.memory_usage(deep=True).sum())
    mem_str = f"{mem/1024:.1f} KB" if mem < 1024**2 else f"{mem/1024**2:.2f} MB"

    return jsonify({
        'filas':              len(df),
        'columnas':           len(df.columns),
        'filas_originales':   len(orig),
        'cant_numericas':     len(cols_num),
        'cant_categoricas':   len(cols_cat),
        'cols_numericas':     cols_num,
        'cols_categoricas':   cols_cat,
        'duplicados':         duplicados,
        'total_faltantes':    total_faltantes,
        'calidad_datos':      calidad_datos,
        'memoria':            mem_str,
        'ranking_faltantes':  ranking_faltantes[:10],
        'info_columnas':      info_columnas,
    })


@rutas_datos.route('/vista_previa', methods=['GET'])
def vista_previa():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    return jsonify(df_a_dict_seguro(obtener_limpio(), 200))


@rutas_datos.route('/limpiar', methods=['POST'])
def limpiar():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    datos   = request.json or {}
    accion  = datos.get('accion')
    df      = obtener_limpio().copy()
    cols_num = columnas_numericas(df)

    if accion == 'eliminar_vacios':
        df = df.dropna()
    elif accion == 'eliminar_duplicados':
        df = df.drop_duplicates()
    elif accion == 'rellenar_media':
        for c in cols_num:
            df[c] = df[c].fillna(df[c].mean())
    elif accion == 'rellenar_mediana':
        for c in cols_num:
            df[c] = df[c].fillna(df[c].median())
    elif accion == 'rellenar_moda':
        for c in df.columns:
            moda = df[c].mode()
            if len(moda):
                df[c] = df[c].fillna(moda.iloc[0])
    elif accion == 'rellenar_personalizado':
        valor   = datos.get('valor', '')
        columna = datos.get('columna')
        destinos = [columna] if columna and columna in df.columns else list(df.columns)
        for c in destinos:
            if pd.api.types.is_numeric_dtype(df[c]):
                try:
                    df[c] = df[c].fillna(float(valor))
                except Exception:
                    pass
            else:
                df[c] = df[c].fillna(str(valor))

    guardar_limpio(df)
    return jsonify({
        'exito':     True,
        'filas':     len(df),
        'columnas':  len(df.columns),
        'faltantes': int(df.isnull().sum().sum()),
    })


@rutas_datos.route('/comparar', methods=['GET'])
def comparar():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400

    def estadisticas(d):
        m  = int(d.isnull().sum().sum())
        tc = d.shape[0] * d.shape[1]
        return {
            'filas':      len(d),
            'columnas':   len(d.columns),
            'faltantes':  m,
            'duplicados': int(d.duplicated().sum()),
            'calidad':    round((1 - m / max(tc, 1)) * 100, 1),
        }

    return jsonify({
        'original': estadisticas(obtener_original()),
        'limpio':   estadisticas(obtener_limpio()),
    })


@rutas_datos.route('/restaurar', methods=['POST'])
def restaurar():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    guardar_limpio(obtener_original().copy())
    return jsonify({'exito': True})
