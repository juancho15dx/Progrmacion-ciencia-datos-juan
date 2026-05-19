"""
Data Manager - Compatible with pandas 3.x
"""
import pandas as pd
import numpy as np

_store = {
    'original': None,
    'clean': None,
    'filename': None,
    'ml_bundle': None,
}

def set_dataset(df, filename=None):
    _store['original'] = df.copy()
    _store['clean'] = df.copy()
    if filename:
        _store['filename'] = filename

def get_original():
    return _store['original']

def get_clean():
    return _store['clean']

def set_clean(df):
    _store['clean'] = df.copy()

def get_filename():
    return _store.get('filename', 'dataset')

def set_ml_bundle(bundle):
    _store['ml_bundle'] = bundle

def get_ml_bundle():
    return _store.get('ml_bundle')

def has_data():
    return _store['original'] is not None

def get_numeric_cols(df):
    """pandas 2 & 3 compatible"""
    return df.select_dtypes(include='number').columns.tolist()

def get_categorical_cols(df):
    """Everything that is NOT numeric"""
    num = set(get_numeric_cols(df))
    return [c for c in df.columns if c not in num]

def safe_val(v):
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
    if pd.isna(v):
        return None
    return v

def df_to_safe_dict(df, max_rows=300):
    df2 = df.head(max_rows).copy()
    cols = list(df2.columns)
    data = []
    for _, row in df2.iterrows():
        r = []
        for v in row:
            try:
                r.append(safe_val(v))
            except Exception:
                r.append(str(v))
        data.append(r)
    return {'columns': cols, 'data': data, 'total_rows': len(df), 'shown_rows': len(df2)}
