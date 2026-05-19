"""
Gestor de Datos — Almacenamiento central del dataset en memoria.
Compatible con pandas 3.x
"""
import pandas as pd
import numpy as np

_almacen = {
    'original':      None,
    'limpio':        None,
    'nombre_archivo': None,
    'modelo_bundle': None,
}

def guardar_dataset(df, nombre_archivo=None):
    _almacen['original'] = df.copy()
    _almacen['limpio']   = df.copy()
    if nombre_archivo:
        _almacen['nombre_archivo'] = nombre_archivo

def obtener_original():
    return _almacen['original']

def obtener_limpio():
    return _almacen['limpio']

def guardar_limpio(df):
    _almacen['limpio'] = df.copy()

def obtener_nombre_archivo():
    return _almacen.get('nombre_archivo', 'dataset')

def guardar_modelo(bundle):
    _almacen['modelo_bundle'] = bundle

def obtener_modelo():
    return _almacen.get('modelo_bundle')

def hay_datos():
    return _almacen['original'] is not None

def columnas_numericas(df):
    """Devuelve columnas numéricas. Funciona con pandas 2 y 3."""
    return df.select_dtypes(include='number').columns.tolist()

def columnas_categoricas(df):
    """Devuelve columnas no numéricas."""
    numericas = set(columnas_numericas(df))
    return [c for c in df.columns if c not in numericas]

def valor_seguro(v):
    if v is None:
        return None
    try:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v

def df_a_dict_seguro(df, max_filas=300):
    """Convierte DataFrame a dict JSON-serializable."""
    df2 = df.head(max_filas).copy()
    columnas = list(df2.columns)
    datos = []
    for _, fila in df2.iterrows():
        fila_lista = []
        for v in fila:
            try:
                fila_lista.append(valor_seguro(v))
            except Exception:
                fila_lista.append(str(v))
        datos.append(fila_lista)
    return {
        'columnas':     columnas,
        'datos':        datos,
        'total_filas':  len(df),
        'filas_mostradas': len(df2),
    }
