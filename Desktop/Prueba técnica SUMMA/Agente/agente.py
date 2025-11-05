import os
import pandas as pd
import json 
from openai import OpenAI

# --- CONFIGURACIÓN GLOBAL Y MANEJO DE ARCHIVOS ---

# RUTA DONDE EL USUARIO GUARDA SU CLAVE API (DEBE SER VÁLIDA PARA EL ACCESO)
API_KEY_FILEPATH = r"C:\Users\Acer\Desktop\GEN AI\Udemy\OPEN AI y API key\API key.txt"

# RUTA DEL ARCHIVO DE DATOS (YA CORREGIDA PARA EL FORMATO XLSX)
DATA_FILE = r"C:\Users\Acer\Desktop\Prueba técnica SUMMA\cesancias_causadas.xlsx" 

MODEL_NAME = "gpt-4-turbo"
LLM_READY = False
TOOL_MAP = {} # Se mapea más abajo

def load_api_key_from_file(filepath: str) -> str:
    """Lee la clave API de OpenAI desde el archivo de texto en la ruta especificada."""
    try:
        with open(filepath, 'r') as f:
            api_key = f.readline().strip()
            if not api_key:
                raise ValueError("El archivo de clave API está vacío.")
            return api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: El archivo de clave API no se encontró en la ruta: {filepath}")
    except Exception as e:
        raise Exception(f"Error al leer el archivo de clave API: {e}")


# --- DEFINICIÓN DE HERRAMIENTAS (TOOL USE / FUNCTION CALLING - MCPs) ---

def get_severance_pay_info(document_id: int) -> str:

    try:
        df = pd.read_excel(DATA_FILE) 
        
        df['Documento'] = pd.to_numeric(df['Documento'], errors='coerce').fillna(0).astype(int)
        record = df[df['Documento'] == document_id]

        if record.empty:
            return f"Error: No se encontró información de cesantías para el Documento {document_id}."

        amount = record['Censatias'].iloc[0]
        month = record['Mes'].iloc[0]
        
        # Formateo del monto
        amount_formatted = f"${amount:,.0f}".replace(",", "_TEMP_").replace(".", ",").replace("_TEMP_", ".")
        
        return f"Información encontrada para Documento {document_id}:\nCesantías Causadas: {amount_formatted}\nMes de Causa: {month}"

    except FileNotFoundError:
        return f"Error: Archivo de datos '{DATA_FILE}' no encontrado. Verifique la ruta absoluta."
    except Exception as e:
        return f"Error al procesar la búsqueda en el archivo: {e}"

# Esquema de la función para que el LLM entienda cómo y cuándo usarla
TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "get_severance_pay_info",
        "description": "Busca la información de cesantías causadas (monto y mes) para un empleado específico, utilizando su número de documento (ID). Útil para responder preguntas sobre datos privados.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "integer",
                    "description": "El número de documento de identificación del empleado (por ejemplo, 131422)."
                }
            },
            "required": ["document_id"]
        }
    }
}]

# Mapeo de la función para que la lógica del agente pueda llamarla
TOOL_MAP["get_severance_pay_info"] = get_severance_pay_info


# LÓGICA DEL AGENTE (EL ORQUESTADOR DE MCPs)
def run_hr_agent(user_question: str) -> str:
    """
    Función principal del agente. Decide si responde con conocimiento base (Opción a) o
    usa la herramienta para acceder a datos (Opción b), implementando la lógica de MCPs.
    """
    global LLM_READY # Necesario para leer el estado de inicialización
    
    if not LLM_READY:
        # Modo Teórico si la clave API falla
        if "documento" in user_question.lower() or "id" in user_question.lower():
            return f"--- PROCESO TEÓRICO: El Agente decidiría que esta pregunta se responde con la Opción B (Acceso a Datos) y llamaría a la función 'get_severance_pay_info'."
        else:
            return f"--- PROCESO TEÓRICO: El Agente decidiría que esta pregunta se responde con la Opción A (Conocimiento Base) sin usar herramientas."

    # 1. Definir la System Instruction y la Mensajería Inicial
    system_instruction = (
        "Eres un agente de soporte de Gestión Humana de una compañía en Colombia. "
        "Tus respuestas deben ser concisas, profesionales y amigables. "
        "Tu objetivo es asistir al usuario con información de cesantías. "
        "Si la pregunta requiere información de cesantías de un empleado específico usando su número de documento (ID), "
        "utiliza **obligatoriamente** la herramienta 'get_severance_pay_info' (Opción b) con el ID extraído de la pregunta."
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_question}
    ]
    
    # 2. Primera Llamada al LLM: ¿Respuesta o Llamada a Función?
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOL_SCHEMA,
        tool_choice="auto" 
    )

    response_message = response.choices[0].message
    
    # 3. Ciclo de Ejecución de la Herramienta (MCPs)
    if response_message.tool_calls:
        # El modelo solicitó usar la herramienta (Opción b - Acceso a Datos)
        
        messages.append(response_message)
        
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            
            if function_name in TOOL_MAP:
                function_args = json.loads(tool_call.function.arguments)
                doc_id = function_args.get('document_id')
                
                print(f"[PROCESO DE PENSAMIENTO]: El agente decidió usar conocimiento específico para el Documento ID: {doc_id}.")
                
                # Ejecutar la función Python real (Acción de la Herramienta)
                tool_output = TOOL_MAP[function_name](document_id=doc_id)
                
                # Formateo de salida para la consola
                tool_output_oneline = tool_output.replace('\n', ' | ')
                print(f"[RESULTADO DE FUNCIÓN]: {tool_output_oneline}")

                # Enviar el resultado de la función de vuelta al LLM
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_output,
                    }
                )
        
        # 4. Segunda Llamada al LLM: Generar la respuesta final con el resultado de la herramienta
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        return response.choices[0].message.content
    
    # Si no hay llamada a función, el modelo responde directamente (Opción a - Conocimiento Base)
    print("\n[PROCESO DE PENSAMIENTO]: Agente decidió responder con Conocimiento Base.")
    return response_message.content


# --- FUNCIÓN DE INICIO DE LA APLICACIÓN DE CONSOLA ---
if __name__ == "__main__":
    
    # Intento de inicialización del cliente al inicio
    try:
        api_key = load_api_key_from_file(API_KEY_FILEPATH)
        llm_client = OpenAI(api_key=api_key)
        LLM_READY = True
        print("=" * 70)
        print("✅ Cliente de OpenAI inicializado. Agente en modo 'Online'.")
    except Exception as e:
        print("=" * 70)
        print(f"⚠️ Advertencia: Error al inicializar el cliente OpenAI. El Agente está en modo 'Teórico'.")
        print(f"Detalle: {e}")
        
    print(f"AGENTE DE GESTIÓN HUMANA | Modelo: {MODEL_NAME}")
    print("Escribe tu pregunta o 'salir' para terminar.")
    print("=" * 70)

    # Bucle interactivo
    while True:
        user_input = input("TÚ: ")
        
        if user_input.lower() in ["salir", "exit"]:
            print("Agente finalizado. ¡Hasta pronto!")
            break
        
        if not user_input.strip():
            continue

        try:
            final_response = run_hr_agent(user_input)
            print("-" * 50)
            print(f"AGENTE: {final_response}")
            print("-" * 50)

        except Exception as e:
            print(f"\n[ERROR CRÍTICO]: No se pudo procesar la pregunta. Detalle: {e}")
            print("-" * 50)
