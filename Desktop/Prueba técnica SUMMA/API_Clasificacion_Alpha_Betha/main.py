import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from model import ClassifierModel 
from typing import Literal, List
import pandas as pd
import io

# ------------------------------------------------
# 1. Definición del Esquema de Datos (Input JSON)
# ------------------------------------------------

# Usamos Pydantic para definir y validar la estructura exacta de la solicitud POST
class FeatureRequest(BaseModel):
    # Numéricas
    Charges: float = Field(..., description="Cargos mensuales (Monthly Charges)", example=70.35)
    Demand: float = Field(..., description="Demanda total del cliente (Total Demand/Revenue)", example=503.6)
    
    # Binarias/Categóricas
    SeniorCity: Literal[0, 1] = Field(..., description="1 si es cliente senior, 0 si no.", example=0)
    Partner: Literal["Yes", "No"] = Field(..., description="Si tiene pareja.", example="Yes")
    Dependents: Literal["Yes", "No"] = Field(..., description="Si tiene dependientes.", example="No")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Facturación sin papel.", example="Yes")
    
    # Categóricas Multi-opción
    Service1: Literal["Yes", "No"] = Field(..., description="Si tiene Servicio 1.", example="Yes")
    Service2: Literal["Yes", "No"] = Field(..., description="Si tiene Servicio 2.", example="No")
    Security: Literal["Yes", "No", "No internet service"] = Field(..., description="Si tiene Seguridad.", example="No")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., description="Si tiene Copia de seguridad online.", example="Yes")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., description="Si tiene Protección de dispositivos.", example="No")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., description="Si tiene Soporte técnico.", example="No")
    
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Tipo de contrato.", example="Month-to-month")
    
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(..., description="Método de pago.", example="Electronic check")


# ------------------------------------------------
# 2. Inicialización de la API y el Modelo
# ------------------------------------------------

app = FastAPI(
    title="API de Clasificación Alpha/Betha",
    description="API para clasificar clientes en Alpha o Betha."
)

# Inicializamos el modelo (carga el PKL) al inicio del servidor
try:
    classifier = ClassifierModel()
except Exception as e:
    print(f"Error fatal al inicializar el modelo: {e}")
    classifier = None 

# ------------------------------------------------
# 3. Definición de Endpoints
# ------------------------------------------------

def check_model_ready():
    """Verifica si el modelo fue cargado correctamente."""
    if classifier is None:
        raise HTTPException(
            status_code=503, 
            detail="El modelo no se cargó correctamente al inicio. Servicio no disponible."
        )

@app.post("/predict")
async def predict_class(features: FeatureRequest):
    """
    Endpoint para clasificación de un registro individual (usado por el cliente de prueba).
    Recibe los datos del formulario, realiza el preprocesamiento,
    clasifica y devuelve la probabilidad y la clase final.
    """
    check_model_ready()
    
    # Convertir el objeto validado por Pydantic a diccionario
    data_json = features.model_dump()
    
    # Llamar a la lógica de clasificación de model.py
    result = classifier.preprocess_and_predict(data_json)
    
    if result is None:
        raise HTTPException(
            status_code=400, 
            detail="Datos incompletos o inválidos. Verifique que todas las variables estén presentes y sean válidas."
        )

    return result

@app.post("/classify_file")
async def classify_file(file: UploadFile = File(..., description="Archivo de Excel (.xlsx) o CSV (.csv) con datos a clasificar.")):
    """
    Endpoint para clasificación por lotes.
    Recibe un archivo (Excel o CSV), lo clasifica y devuelve los resultados en formato CSV para descarga.
    """
    check_model_ready()

    # Manejo de la carga del archivo
    try:
        # Leer el contenido binario del archivo en memoria
        content = await file.read()
        
        # Determinar el tipo de archivo y leerlo con pandas
        if file.filename.endswith('.xlsx'):
            df_input = pd.read_excel(io.BytesIO(content), engine='openpyxl')
        elif file.filename.endswith('.csv'):
            # Asume que el CSV usa coma o punto y coma como separador y lo intenta adivinar
            try:
                df_input = pd.read_csv(io.BytesIO(content), encoding='utf-8')
            except UnicodeDecodeError:
                df_input = pd.read_csv(io.BytesIO(content), encoding='latin-1')
        else:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Use .xlsx o .csv.")
        
        # Omitir cualquier columna que el modelo no necesite o que pueda causar problemas (ID o Class original)
        df_input = df_input.drop(columns=['autoID', 'Class'], errors='ignore')
        
        results = []
        
        # Iterar sobre cada fila para clasificar
        for index, row in df_input.iterrows():
            # Convertir la fila (Serie) a diccionario
            data_json = row.to_dict()
            
            # Asegurarse de que SeniorCity sea entero si existe, para cumplir con Pydantic
            if 'SeniorCity' in data_json:
                data_json['SeniorCity'] = int(data_json['SeniorCity'])
                
            # Llamar a la función de clasificación del modelo
            result = classifier.preprocess_and_predict(data_json)
            
            # Si el resultado es None (datos inválidos), registrar el error pero no detener
            if result is not None:
                # Combinar datos originales con la predicción
                result_row = {**row.to_dict(), **result}
                results.append(result_row)
            else:
                # Opcional: Agregar una fila de error para datos inválidos
                error_row = {**row.to_dict(), "Error_Clasificacion": "Datos de entrada inválidos o incompletos"}
                results.append(error_row)

        if not results:
            raise HTTPException(status_code=404, detail="No se pudo clasificar ningún registro válido en el archivo.")

        # Crear un DataFrame con los resultados
        df_results = pd.DataFrame(results)
        
        # --- Preparar la Respuesta como CSV para Descarga ---
        # Usar io.StringIO para escribir el CSV en memoria
        output = io.StringIO()
        df_results.to_csv(output, index=False)
        
        # Usar io.BytesIO para manejar la codificación de forma segura
        output.seek(0)
        output_bytes = io.BytesIO(output.read().encode('utf-8'))
        
        # Devolver el archivo CSV usando StreamingResponse
        return StreamingResponse(
            output_bytes,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=resultados_clasificacion.csv"
            }
        )

    except HTTPException as e:
        # Re-lanzar errores HTTP específicos (como 400 por formato no soportado)
        raise e     
    except Exception as e:
        # Manejo general de errores
        print(f"Error durante el procesamiento del archivo: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al procesar el archivo: {e}")


# ------------------------------------------------
# 4. Configuración de Ejecución
# ------------------------------------------------

if __name__ == "__main__":
    # Comando para iniciar el servidor Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
