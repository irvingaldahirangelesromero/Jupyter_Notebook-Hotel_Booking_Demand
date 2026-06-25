"""
Aplicación Flask para el despliegue del modelo de predicción de tarifa diaria (adr)
del proyecto de regresión CRISP-DM — Hotel Booking Demand.

Carga:
  - modelo_adr_pipeline.joblib  -> Pipeline completo (ColumnTransformer + PCA + MLPRegressor)
  - metadata_despliegue.json    -> columnas numéricas/categóricas y categorías reales
                                    vistas por el OneHotEncoder durante el entrenamiento.

"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ------------------------------------------------------------------
# Carga del pipeline entrenado y de la metadata de columnas/categorías
# ------------------------------------------------------------------
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(RUTA_BASE, "modelo_adr_pipeline.joblib")
RUTA_METADATA = os.path.join(RUTA_BASE, "metadata_despliegue.json")

pipeline_modelo = joblib.load(RUTA_MODELO)

with open(RUTA_METADATA, "r", encoding="utf-8") as f:
    metadata = json.load(f)

NUMERIC_COLS = metadata["numeric_cols"]
CATEGORICAL_COLS = metadata["categorical_cols"]
ORDEN_COLUMNAS_X = metadata["orden_columnas_X"]
CATEGORIAS = metadata["categorias_categoricas"]

# Etiquetas legibles para los campos numéricos (solo presentación en el formulario)
ETIQUETAS_NUMERICAS = {
    "lead_time": "Días de anticipación de la reserva (lead_time)",
    "arrival_date_year": "Año de llegada",
    "arrival_date_week_number": "Número de semana de llegada (1-53)",
    "arrival_date_day_of_month": "Día del mes de llegada (1-31)",
    "stays_in_weekend_nights": "Noches de fin de semana",
    "stays_in_week_nights": "Noches entre semana",
    "adults": "Número de adultos",
    "children": "Número de niños",
    "babies": "Número de bebés",
    "is_repeated_guest": "¿Huésped repetido? (0 = No, 1 = Sí)",
    "previous_cancellations": "Cancelaciones previas",
    "previous_bookings_not_canceled": "Reservas previas no canceladas",
    "booking_changes": "Cambios realizados a la reserva",
    "days_in_waiting_list": "Días en lista de espera",
    "required_car_parking_spaces": "Espacios de estacionamiento requeridos",
    "total_of_special_requests": "Número de peticiones especiales",
}

# Etiquetas legibles para los campos categóricos
ETIQUETAS_CATEGORICAS = {
    "hotel": "Tipo de hotel",
    "arrival_date_month": "Mes de llegada",
    "meal": "Tipo de plan de comidas",
    "country": "País de origen",
    "market_segment": "Segmento de mercado",
    "distribution_channel": "Canal de distribución",
    "reserved_room_type": "Tipo de habitación reservada",
    "deposit_type": "Tipo de depósito",
    "customer_type": "Tipo de cliente",
}


def construir_campos_formulario():
    """Arma la lista de campos (en el mismo orden que X) que renderiza la plantilla."""
    campos = []
    for columna in ORDEN_COLUMNAS_X:
        if columna in NUMERIC_COLS:
            campos.append({
                "nombre": columna,
                "tipo": "numero",
                "etiqueta": ETIQUETAS_NUMERICAS.get(columna, columna),
            })
        else:
            campos.append({
                "nombre": columna,
                "tipo": "categoria",
                "etiqueta": ETIQUETAS_CATEGORICAS.get(columna, columna),
                "opciones": CATEGORIAS.get(columna, []),
            })
    return campos


@app.route("/", methods=["GET"])
def index():
    campos = construir_campos_formulario()
    return render_template("index.html", campos=campos, prediccion=None, error=None)


@app.route("/predecir", methods=["POST"])
def predecir():
    campos = construir_campos_formulario()
    try:
        datos_formulario = {}

        for columna in NUMERIC_COLS:
            valor_texto = request.form.get(columna, "").strip()
            if valor_texto == "":
                raise ValueError(f"El campo numérico '{columna}' está vacío.")
            datos_formulario[columna] = float(valor_texto)
#  k
        for columna in CATEGORICAL_COLS:
            valor_texto = request.form.get(columna, "").strip()
            if valor_texto == "":
                raise ValueError(f"El campo '{columna}' no fue seleccionado.")
            datos_formulario[columna] = valor_texto

        # Se construye un DataFrame de una sola fila con exactamente las mismas
        # columnas (y el mismo orden) que vio el pipeline durante el entrenamiento.
        fila_entrada = pd.DataFrame([datos_formulario], columns=ORDEN_COLUMNAS_X)

        prediccion = float(pipeline_modelo.predict(fila_entrada)[0])
        prediccion = round(prediccion, 2)

        return render_template("index.html", campos=campos, prediccion=prediccion, error=None)

    except Exception as exc:
        return render_template("index.html", campos=campos, prediccion=None, error=str(exc))


if __name__ == "__main__":
    # Para ejecución local. En Render, el servidor de producción (gunicorn) ya no usa este bloque.
    app.run(debug=True, host="0.0.0.0", port=5000)
