"""Rutas de Exportación — Excel y PDF."""
from flask import Blueprint, request, jsonify, send_file
import pandas as pd
import os, sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from modules.gestor_datos import obtener_limpio, obtener_original, hay_datos, columnas_numericas, columnas_categoricas

rutas_exportacion = Blueprint('exportacion', __name__)
CARPETA_EXPORTACION = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'exports')


@rutas_exportacion.route('/excel', methods=['POST'])
def exportar_excel():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df       = obtener_limpio()
    original = obtener_original()
    cols_num = columnas_numericas(df)
    cols_cat = columnas_categoricas(df)
    os.makedirs(CARPETA_EXPORTACION, exist_ok=True)
    sello   = datetime.now().strftime('%Y%m%d_%H%M%S')
    ruta    = os.path.join(CARPETA_EXPORTACION, f'BI_Reporte_{sello}.xlsx')
    try:
        with pd.ExcelWriter(ruta, engine='openpyxl') as escritor:
            df.to_excel(escritor, sheet_name='Datos_Limpios',    index=False)
            original.to_excel(escritor, sheet_name='Datos_Originales', index=False)
            if cols_num:
                est = df[cols_num].describe(percentiles=[.1, .25, .5, .75, .9]).T
                try:
                    est['asimetria'] = df[cols_num].skew()
                    est['curtosis']  = df[cols_num].kurtosis()
                except Exception:
                    pass
                est.reset_index().rename(columns={'index': 'variable'}).to_excel(
                    escritor, sheet_name='Estadísticas', index=False)
            if len(cols_num) >= 2:
                df[cols_num].corr().round(4).to_excel(escritor, sheet_name='Correlaciones')
            if cols_cat:
                filas = []
                for col in cols_cat:
                    vc = df[col].astype(str).value_counts()
                    for cat, cnt in vc.items():
                        filas.append({
                            'Columna': col, 'Categoría': str(cat),
                            'Frecuencia': int(cnt),
                            'Porcentaje_%': round(cnt / max(len(df), 1) * 100, 2),
                        })
                pd.DataFrame(filas).to_excel(escritor, sheet_name='Frecuencias', index=False)
            faltantes = df.isnull().sum().reset_index()
            faltantes.columns = ['Columna', 'Valores_Faltantes']
            faltantes['Porcentaje_%'] = (faltantes['Valores_Faltantes'] / max(len(df), 1) * 100).round(2)
            faltantes.to_excel(escritor, sheet_name='Valores_Faltantes', index=False)
            try:
                from openpyxl.styles import PatternFill, Font, Alignment
                relleno_cab = PatternFill(start_color='1E3A5F', end_color='1E3A5F', fill_type='solid')
                fuente_cab  = Font(color='FFFFFF', bold=True)
                for hoja in escritor.book.worksheets:
                    for celda in hoja[1]:
                        celda.fill       = relleno_cab
                        celda.font       = fuente_cab
                        celda.alignment  = Alignment(horizontal='center')
                    for col_celdas in hoja.columns:
                        ancho = max((len(str(c.value or '')) for c in col_celdas), default=8)
                        hoja.column_dimensions[col_celdas[0].column_letter].width = min(ancho + 3, 40)
            except Exception:
                pass
        return send_file(ruta, as_attachment=True,
                         download_name=f'BI_Reporte_{sello}.xlsx',
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@rutas_exportacion.route('/pdf', methods=['POST'])
def exportar_pdf():
    if not hay_datos():
        return jsonify({'error': 'No hay dataset cargado'}), 400
    df       = obtener_limpio()
    cols_num = columnas_numericas(df)
    cols_cat = columnas_categoricas(df)
    os.makedirs(CARPETA_EXPORTACION, exist_ok=True)
    sello = datetime.now().strftime('%Y%m%d_%H%M%S')
    ruta  = os.path.join(CARPETA_EXPORTACION, f'BI_Reporte_{sello}.pdf')
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
        from reportlab.lib.enums import TA_CENTER

        doc    = SimpleDocTemplate(ruta, pagesize=A4,
                                   topMargin=1.5*cm, bottomMargin=1.5*cm,
                                   leftMargin=2*cm, rightMargin=2*cm)
        estilos = getSampleStyleSheet()
        E = lambda nombre, **kw: ParagraphStyle(nombre, parent=estilos['Normal'], **kw)
        titulo_est  = E('T', fontSize=20, textColor=colors.HexColor('#0F172A'),
                        alignment=TA_CENTER, spaceAfter=6, fontName='Helvetica-Bold')
        h1_est      = E('H1', fontSize=13, textColor=colors.HexColor('#1e40af'),
                        spaceBefore=12, spaceAfter=4, fontName='Helvetica-Bold')
        cuerpo_est  = E('C', fontSize=10, spaceAfter=4)

        def crear_tabla(datos, anchos, color_cab='#1e40af', colores_fila=None):
            t = Table(datos, colWidths=anchos)
            cf = colores_fila or [colors.HexColor('#f8fafc'), colors.white]
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(color_cab)),
                ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
                ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE',   (0, 0), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), cf),
                ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('ALIGN',      (1, 0), (-1, -1), 'CENTER'),
                ('PADDING',    (0, 0), (-1, -1), 5),
            ]))
            return t

        historia = [
            Spacer(1, 0.5*cm),
            Paragraph("Reporte de Inteligencia de Negocios", titulo_est),
            Paragraph(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}", cuerpo_est),
            HRFlowable(width='100%', thickness=2, color=colors.HexColor('#1e40af')),
            Spacer(1, 0.4*cm),
        ]

        faltantes_total = int(df.isnull().sum().sum())
        total_celdas    = df.shape[0] * df.shape[1]
        calidad         = round((1 - faltantes_total / max(total_celdas, 1)) * 100, 1)

        historia.append(Paragraph("1. Resumen del Dataset", h1_est))
        historia.append(crear_tabla(
            [['Métrica', 'Valor'],
             ['Total de filas',    str(len(df))],
             ['Total de columnas', str(len(df.columns))],
             ['Variables numéricas',   str(len(cols_num))],
             ['Variables categóricas', str(len(cols_cat))],
             ['Valores faltantes', str(faltantes_total)],
             ['Calidad de datos',  f'{calidad}%']],
            [9*cm, 5*cm]
        ))
        historia.append(Spacer(1, 0.4*cm))

        if cols_num:
            historia.append(Paragraph("2. Estadísticas Descriptivas", h1_est))
            cab  = ['Variable', 'Media', 'Mediana', 'Desv. Std', 'Mínimo', 'Máximo', 'Asimetría']
            filas = [cab] + [
                [col[:18], f"{df[col].mean():.2f}", f"{df[col].median():.2f}",
                 f"{df[col].std():.2f}", f"{df[col].min():.2f}", f"{df[col].max():.2f}",
                 f"{df[col].skew():.3f}"]
                for col in cols_num[:12]
            ]
            historia.append(crear_tabla(filas, [4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm],
                                         '#7c3aed', [colors.HexColor('#f5f3ff'), colors.white]))
            historia.append(Spacer(1, 0.4*cm))

        if cols_cat:
            historia.append(Paragraph("3. Variables Categóricas", h1_est))
            for col in cols_cat[:4]:
                historia.append(Paragraph(f"• {col}", cuerpo_est))
                vc    = df[col].astype(str).value_counts().head(5)
                filas = [['Categoría', 'Frecuencia', 'Porcentaje']] + [
                    [str(k)[:28], str(int(v)), f"{round(v / max(len(df), 1) * 100, 1)}%"]
                    for k, v in vc.items()
                ]
                historia.append(crear_tabla(filas, [8*cm, 4*cm, 4*cm], '#059669',
                                             [colors.HexColor('#f0fdf4'), colors.white]))
                historia.append(Spacer(1, 0.2*cm))

        historia += [
            Spacer(1, 0.8*cm),
            HRFlowable(width='100%', thickness=1, color=colors.HexColor('#e2e8f0')),
            Paragraph("Generado por BI Platform — Plataforma de Inteligencia de Negocios",
                       E('pie', fontSize=8, textColor=colors.grey, alignment=TA_CENTER)),
        ]
        doc.build(historia)
        return send_file(ruta, as_attachment=True,
                         download_name=f'BI_Reporte_{sello}.pdf', mimetype='application/pdf')
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'detalle': traceback.format_exc()}), 500
