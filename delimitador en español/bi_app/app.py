"""
BI Platform — Plataforma de Inteligencia de Negocios
Punto de entrada principal de la aplicación.
"""
import os
import secrets
from flask import Flask, render_template

aplicacion = Flask(__name__)
aplicacion.secret_key = secrets.token_hex(32)
aplicacion.config['UPLOAD_FOLDER']      = os.path.join(os.path.dirname(__file__), 'uploads')
aplicacion.config['EXPORT_FOLDER']      = os.path.join(os.path.dirname(__file__), 'exports')
aplicacion.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

os.makedirs(aplicacion.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(aplicacion.config['EXPORT_FOLDER'], exist_ok=True)

# Registro de rutas
from routes.rutas_datos        import rutas_datos
from routes.rutas_estadisticas import rutas_estadisticas
from routes.rutas_correlacion  import rutas_correlacion
from routes.rutas_cualitativas import rutas_cualitativas
from routes.rutas_ml           import rutas_ml
from routes.rutas_dashboard    import rutas_dashboard
from routes.rutas_exportacion  import rutas_exportacion

aplicacion.register_blueprint(rutas_datos,        url_prefix='/api/datos')
aplicacion.register_blueprint(rutas_estadisticas, url_prefix='/api/estadisticas')
aplicacion.register_blueprint(rutas_correlacion,  url_prefix='/api/correlacion')
aplicacion.register_blueprint(rutas_cualitativas, url_prefix='/api/cualitativas')
aplicacion.register_blueprint(rutas_ml,           url_prefix='/api/ml')
aplicacion.register_blueprint(rutas_dashboard,    url_prefix='/api/dashboard')
aplicacion.register_blueprint(rutas_exportacion,  url_prefix='/api/exportacion')


@aplicacion.route('/')
def inicio():
    return render_template('index.html')


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  BI Platform — Inteligencia de Negocios")
    print("=" * 60)
    print("  Abre tu navegador en: http://localhost:5050")
    print("=" * 60 + "\n")
    aplicacion.run(debug=False, port=5050, host='0.0.0.0')

# Módulo gerencial avanzado
from routes.rutas_gerencial import rutas_gerencial
aplicacion.register_blueprint(rutas_gerencial, url_prefix='/api/gerencial')
