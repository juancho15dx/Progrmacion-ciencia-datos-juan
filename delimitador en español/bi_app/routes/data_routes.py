from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.data_manager import (set_dataset, get_original, get_clean, set_clean,
    has_data, df_to_safe_dict, get_numeric_cols, get_categorical_cols)
from modules.chart_utils import read_any_file, auto_convert_numeric

data_bp = Blueprint('data', __name__)

@data_bp.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    file = request.files['file']
    fname = file.filename or ''
    ext = fname.lower().rsplit('.',1)[-1] if '.' in fname else ''
    if ext not in ('csv','xlsx','xls'):
        return jsonify({'error': 'Solo se admiten archivos CSV o Excel (.xlsx/.xls)'}), 400
    df, err = read_any_file(file)
    if err:
        return jsonify({'error': err}), 400
    df = auto_convert_numeric(df)
    set_dataset(df, fname)
    return jsonify({'success': True, 'filename': fname, 'rows': len(df), 'cols': len(df.columns)})

@data_bp.route('/overview', methods=['GET'])
def overview():
    if not has_data():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df = get_clean()
    orig = get_original()
    num_cols = get_numeric_cols(df)
    cat_cols = get_categorical_cols(df)
    missing = df.isnull().sum()
    missing_pct = (missing / max(len(df),1) * 100).round(2)
    duplicates = int(df.duplicated().sum())
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(missing.sum())
    data_quality = round((1 - total_missing / max(total_cells,1)) * 100, 1)

    missing_rank = sorted(
        [{'column': c, 'missing': int(missing[c]), 'pct': float(missing_pct[c])}
         for c in df.columns if missing[c] > 0],
        key=lambda x: x['missing'], reverse=True)

    col_info = [{'name': c, 'dtype': str(df[c].dtype),
                 'unique': int(df[c].nunique()),
                 'missing': int(missing[c]),
                 'missing_pct': float(missing_pct[c]),
                 'type': 'Numérica' if c in num_cols else 'Categórica'}
                for c in df.columns]

    mem = int(df.memory_usage(deep=True).sum())
    mem_str = f"{mem/1024:.1f} KB" if mem < 1024**2 else f"{mem/1024**2:.2f} MB"

    return jsonify({
        'rows': len(df), 'cols': len(df.columns),
        'orig_rows': len(orig),
        'numeric_count': len(num_cols), 'categorical_count': len(cat_cols),
        'numeric_cols': num_cols, 'categorical_cols': cat_cols,
        'duplicates': duplicates, 'total_missing': total_missing,
        'data_quality': data_quality, 'memory': mem_str,
        'missing_rank': missing_rank[:10], 'col_info': col_info,
    })

@data_bp.route('/preview', methods=['GET'])
def preview():
    if not has_data():
        return jsonify({'error': 'No hay dataset'}), 400
    return jsonify(df_to_safe_dict(get_clean(), 200))

@data_bp.route('/clean', methods=['POST'])
def clean():
    if not has_data():
        return jsonify({'error': 'No hay dataset'}), 400
    data = request.json or {}
    action = data.get('action')
    df = get_clean().copy()
    num_cols = get_numeric_cols(df)

    if action == 'drop_nulls':
        df = df.dropna()
    elif action == 'drop_duplicates':
        df = df.drop_duplicates()
    elif action == 'fill_mean':
        for c in num_cols: df[c] = df[c].fillna(df[c].mean())
    elif action == 'fill_median':
        for c in num_cols: df[c] = df[c].fillna(df[c].median())
    elif action == 'fill_mode':
        for c in df.columns:
            m = df[c].mode()
            if len(m): df[c] = df[c].fillna(m.iloc[0])
    elif action == 'fill_custom':
        value = data.get('value','')
        col = data.get('column')
        targets = [col] if col and col in df.columns else list(df.columns)
        for c in targets:
            if pd.api.types.is_numeric_dtype(df[c]):
                try: df[c] = df[c].fillna(float(value))
                except: pass
            else:
                df[c] = df[c].fillna(str(value))

    set_clean(df)
    return jsonify({'success': True, 'rows': len(df), 'cols': len(df.columns),
                    'missing': int(df.isnull().sum().sum())})

@data_bp.route('/compare', methods=['GET'])
def compare():
    if not has_data():
        return jsonify({'error': 'No hay dataset'}), 400
    def stats(d):
        m = int(d.isnull().sum().sum())
        tc = d.shape[0]*d.shape[1]
        return {'rows': len(d), 'cols': len(d.columns), 'missing': m,
                'duplicates': int(d.duplicated().sum()),
                'quality': round((1-m/max(tc,1))*100,1)}
    return jsonify({'original': stats(get_original()), 'clean': stats(get_clean())})

@data_bp.route('/reset', methods=['POST'])
def reset():
    if not has_data():
        return jsonify({'error': 'No hay dataset'}), 400
    set_clean(get_original().copy())
    return jsonify({'success': True})
