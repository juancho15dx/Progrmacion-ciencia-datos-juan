from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys, base64, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.data_manager import get_clean, has_data, get_numeric_cols, get_categorical_cols, set_ml_bundle, get_ml_bundle
from modules.chart_utils import fig_json, DARK_LAYOUT, PALETTE, read_any_file

ml_bp = Blueprint('ml', __name__)

@ml_bp.route('/columns', methods=['GET'])
def get_columns():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    df = get_clean()
    return jsonify({'all_columns': df.columns.tolist(),
                    'numeric': get_numeric_cols(df),
                    'categorical': get_categorical_cols(df)})

@ml_bp.route('/train', methods=['POST'])
def train():
    if not has_data(): return jsonify({'error': 'No hay dataset'}), 400
    data = request.json or {}
    target = data.get('target'); features = data.get('features',[]); ml_type = data.get('ml_type','regression')
    if not target or not features:
        return jsonify({'error': 'Selecciona variable objetivo y predictoras'}), 400

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix

    df = get_clean().copy().dropna(subset=[target])
    if len(df) < 10:
        return jsonify({'error': 'Insuficientes datos (mínimo 10 filas sin nulos en objetivo)'}), 400

    valid_feats = [f for f in features if f in df.columns and f != target]
    if not valid_feats:
        return jsonify({'error': 'Sin variables predictoras válidas'}), 400

    X = df[valid_feats].copy()
    y = df[target].copy()

    label_encoders = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna('_NA_'))
            label_encoders[col] = le
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())

    target_encoder = None; classes = None
    if ml_type == 'classification':
        target_encoder = LabelEncoder()
        y_enc = target_encoder.fit_transform(y.astype(str))
        classes = target_encoder.classes_.tolist()
    else:
        y_enc = pd.to_numeric(y, errors='coerce')
        mask = y_enc.notna(); X = X[mask]; y_enc = y_enc[mask]

    if len(X) < 10:
        return jsonify({'error': 'Insuficientes datos tras conversión numérica'}), 400

    X_arr = X.values; y_arr = y_enc.values if hasattr(y_enc,'values') else np.array(y_enc)
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)

    metrics = {}; pred_fig = conf_fig = imp_fig = None; importance = []

    if ml_type == 'regression':
        model = LinearRegression()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        metrics = {'rmse': round(float(np.sqrt(mean_squared_error(y_test,y_pred))),4),
                   'mae': round(float(mean_absolute_error(y_test,y_pred)),4),
                   'r2': round(float(r2_score(y_test,y_pred)),4)}
        importance = sorted([{'feature':f,'importance':round(float(c),4)} for f,c in zip(valid_feats,model.coef_)],
                             key=lambda x: abs(x['importance']), reverse=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.tolist(), y=y_pred.tolist(), mode='markers',
                                 marker=dict(color='#00d4ff',size=6,opacity=0.65), name='Predicciones'))
        mn,mx = float(min(y_test.min(),y_pred.min())), float(max(y_test.max(),y_pred.max()))
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                                 line=dict(color='#f59e0b',dash='dash',width=2), name='Ideal'))
        fig.update_layout(**DARK_LAYOUT, title='Real vs Predicho', xaxis_title='Real', yaxis_title='Predicho', height=380)
        pred_fig = fig_json(fig)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        acc = float(accuracy_score(y_test,y_pred))
        metrics = {'accuracy': round(acc,4)}
        cm = confusion_matrix(y_test,y_pred)
        cls_labels = [str(c) for c in (classes or sorted(set(y_test.tolist())))]
        fig_cm = go.Figure(go.Heatmap(z=cm.tolist(), x=cls_labels, y=cls_labels,
                                      colorscale=[[0,'#1e293b'],[1,'#00d4ff']],
                                      text=cm.tolist(), texttemplate='%{text}',
                                      textfont=dict(size=12, color='white')))
        fig_cm.update_layout(**DARK_LAYOUT, title='Matriz de Confusión', height=380)
        conf_fig = fig_json(fig_cm)
        if hasattr(model,'coef_'):
            coef = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim>1 else np.abs(model.coef_[0])
            importance = sorted([{'feature':f,'importance':round(float(c),4)} for f,c in zip(valid_feats,coef)],
                                 key=lambda x: abs(x['importance']), reverse=True)

    if importance:
        fig_i = go.Figure(go.Bar(x=[i['importance'] for i in importance],
                                  y=[i['feature'] for i in importance],
                                  orientation='h', marker=dict(color=PALETTE[:len(importance)]),
                                  text=[str(i['importance']) for i in importance], textposition='outside'))
        fig_i.update_layout(**DARK_LAYOUT, title='Importancia de Variables',
                             height=max(320, len(importance)*36+100))
        imp_fig = fig_json(fig_i)

    bundle = {'model':model,'scaler':scaler,'features':valid_feats,'target':target,
              'ml_type':ml_type,'target_encoder':target_encoder,'label_encoders':label_encoders}
    set_ml_bundle(bundle)
    model_b64 = base64.b64encode(pickle.dumps(bundle)).decode('utf-8')

    return jsonify({'success':True,'metrics':metrics,'importance':importance[:15],
                    'pred_fig':pred_fig,'conf_fig':conf_fig,'imp_fig':imp_fig,
                    'model_b64':model_b64,'train_samples':len(X_train),'test_samples':len(X_test),
                    'classes':classes})

@ml_bp.route('/predict', methods=['POST'])
def predict():
    bundle = get_ml_bundle()
    if bundle is None: return jsonify({'error': 'No hay modelo entrenado'}), 400
    if 'file' not in request.files: return jsonify({'error': 'No se envió archivo'}), 400
    file = request.files['file']
    df_pred, err = read_any_file(file)
    if err: return jsonify({'error': err}), 400

    try:
        model = bundle['model']; scaler = bundle['scaler']; features = bundle['features']
        ml_type = bundle['ml_type']; te = bundle.get('target_encoder'); les = bundle.get('label_encoders',{})

        missing_feats = [f for f in features if f not in df_pred.columns]
        if missing_feats:
            return jsonify({'error': f'Columnas faltantes: {", ".join(missing_feats)}'}), 400

        X = df_pred[features].copy()
        for col in X.columns:
            if col in les:
                X[col] = les[col].transform(X[col].astype(str).fillna('_NA_'))
            elif not pd.api.types.is_numeric_dtype(X[col]):
                from sklearn.preprocessing import LabelEncoder
                le2 = LabelEncoder(); X[col] = le2.fit_transform(X[col].astype(str).fillna('_NA_'))
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        X_s = scaler.transform(X.values)
        preds = model.predict(X_s)
        preds_label = te.inverse_transform(preds) if (ml_type=='classification' and te) else preds

        df_result = df_pred.copy(); df_result['PREDICCION'] = preds_label

        if ml_type == 'classification':
            from collections import Counter
            cnt = Counter(preds_label.tolist() if hasattr(preds_label,'tolist') else list(preds_label))
            fig = go.Figure(go.Bar(x=list(cnt.keys()), y=list(cnt.values()),
                                   marker=dict(color=PALETTE[:len(cnt)])))
        else:
            fig = go.Figure(go.Histogram(x=preds_label.tolist() if hasattr(preds_label,'tolist') else list(preds_label),
                                         marker_color='#7c3aed', nbinsx=30))
        fig.update_layout(**DARK_LAYOUT, title='Distribución de Predicciones', height=350)

        cols = df_result.columns.tolist()
        data = [['' if pd.isna(v) else str(v) for v in row] for _,row in df_result.head(100).iterrows()]
        return jsonify({'success':True,'total_predictions':len(preds_label),
                        'preview':{'columns':cols,'data':data,'total':len(df_result)},
                        'fig':fig_json(fig)})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500
