import requests
import pandas as pd
import json

# URL del endpoint de tu API (Tu API está corriendo aquí)
API_URL = "http://127.0.0.1:8000/predict"
# Nombre del archivo con los datos a predecir
INPUT_FILE = "to_predict.csv"
# Nombre del archivo de salida requerido
OUTPUT_FILE = "results_ClasificacionAlphaBetha.csv"

def run_predictions():
    """
    Lee los datos del CSV, hace peticiones POST a la API y guarda los resultados.
    """
    try:
        # 1. Cargar los datos a predecir
        print(f"Cargando datos desde {INPUT_FILE}...")
        # Ignoramos la columna 'autoID' y 'Class' si existen, ya que la API solo espera las características
        # >>> CAMBIO AQUÍ: Se especifica el separador como punto y coma (;) <<<
        df_input = pd.read_csv(INPUT_FILE, sep=';').drop(columns=['autoID', 'Class'], errors='ignore')
        
        results = []
        
        # 2. Iterar sobre cada fila y enviar la petición
        for index, row in df_input.iterrows():
            # Convertir la fila (Serie de Pandas) a un diccionario JSON para la API
            data_json = row.to_dict()
            
            # Asegurarse de que el campo SeniorCity sea int (0 o 1) como lo espera Pydantic
            if 'SeniorCity' in data_json:
                # Se asegura la conversión a entero, que ahora es posible porque SeniorCity es una columna individual
                data_json['SeniorCity'] = int(data_json['SeniorCity'])
                
            print(f"\n-> Enviando petición para el registro {index + 1}...")

            # Realizar la petición POST
            try:
                response = requests.post(API_URL, json=data_json)
                response.raise_for_status() 

                # Obtener la respuesta JSON de la API
                api_result = response.json()
                
                # Combinar los datos de entrada con la predicción
                result_row = {**row.to_dict(), **api_result}
                results.append(result_row)
                
                print(f"   Resultado: {api_result['Class']} | Probabilidad Alpha: {api_result['Probabilidad_Alpha']:.4f}")

            except requests.exceptions.HTTPError as e:
                print(f"   ❌ ERROR HTTP: {e}. Respuesta API: {response.text}")
            except requests.exceptions.ConnectionError:
                print(f"   ❌ ERROR de Conexión: Asegúrese de que la API esté corriendo en {API_URL}.")
                return

        # 3. Guardar los resultados
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(OUTPUT_FILE, index=False)
            print(f"\n✅ Proceso completado. Resultados guardados en: {OUTPUT_FILE}")
        else:
            print("\n⚠️ No se pudieron obtener resultados para guardar.")
            
    except FileNotFoundError:
        print(f"❌ ERROR: El archivo de entrada '{INPUT_FILE}' no se encontró.")
    except Exception as e:
        print(f"❌ ERROR inesperado en el cliente: {e}")


if __name__ == "__main__":
    run_predictions()
