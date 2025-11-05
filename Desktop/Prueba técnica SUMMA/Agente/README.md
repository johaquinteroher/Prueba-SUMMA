Agente de Soporte de Gestión Humana (RRHH)

Este proyecto implementa un Agente de Inteligencia Artificial de Soporte para Gestión Humana que opera bajo la arquitectura de Cadenas de Pensamiento Múltiple (MCPs - Multiple Chain of Thoughts).

El objetivo principal es delegar al Agente la responsabilidad de decidir la mejor ruta para responder una consulta del usuario:

Ruta A (Conocimiento Base): Para preguntas generales de política o proceso (ej: ¿Qué debo hacer para solicitar cesantias?).

Ruta B (Acceso a Datos/Herramientas): Para preguntas que requieren acceso a información privada o en bases de datos (ej: ¿Cuál es el monto de cesantías del empleado 131422?).

Arquitectura y Funcionamiento (MCPs)

El agente opera en dos pasos de pensamiento crítico para tomar decisiones:

Paso 1: Decisión del LLM (Tool Use)

El usuario realiza una pregunta al Agente. El Modelo de Lenguaje Grande (LLM), gracias a su "System Instruction" y la definición del esquema de la herramienta (TOOL_SCHEMA), evalúa la pregunta:

Si la pregunta es de política/general: Responde inmediatamente con la información que tiene en su conocimiento interno (Ruta A).

Si la pregunta contiene un ID de Documento: Genera automáticamente una llamada a la función Python get_severance_pay_info (Ruta B).

Paso 2: Ejecución de la Herramienta (Función Externa)

Si el LLM solicitó la Ruta B, el código Python toma el control y ejecuta la función real, pasándole el ID extraído por el modelo:

Función Ejecutada: get_severance_pay_info(document_id).

Acción: La función utiliza la librería pandas para leer el archivo local cesancias_causadas.xlsx.

Resultado: Busca el ID y devuelve la información (Monto y Mes de Causa) en formato de texto.

Paso 3: Respuesta Final Generada

El resultado de la función se envía de vuelta al LLM como parte del contexto. El Agente ahora tiene la información necesaria y formula una respuesta final, concisa y profesional, para el usuario.

⚙️ Configuración del Proyecto

1. Requisitos Previos

Asegúrate de tener instalado Python 3.x y haber creado y activado un entorno virtual.

2. Archivos Necesarios

El proyecto requiere tres archivos para su ejecución:

Archivo

Descripción

Uso en el Agente

agente_interactivo.py ---> El script principal de la aplicación de consola-->Contiene toda la lógica del Agente y la función de herramienta.

requirements.txt--> Lista las dependencias de Python----> Instala openai, pandas y openpyxl entre otras librerias


cesancias_causadas.xlsx --> El archivo de datos de cesantías.-->La fuente de datos a la que la función get_severance_pay_info accede.


3. Instalación de Dependencias

Utiliza el archivo requirements.txt para instalar todas las librerías necesarias.

pip install -r requirements.txt


4. Configuración de la API y Ruta de Datos

Antes de ejecutar, debes editar dos variables en la parte superior del archivo agente_interactivo.py:

API_KEY_FILEPATH: Debe apuntar a la ruta absoluta de tu archivo de texto que contiene la clave de OpenAI (GPT-4-Turbo).

API_KEY_FILEPATH = r"C:\ruta\a\tu\API key.txt"


DATA_FILE: Debe apuntar a la ruta absoluta donde se encuentra el archivo de Excel.

DATA_FILE = r"C:\Users\Acer\Desktop\Prueba técnica SUMMA\cesancias_causadas.xlsx" 


5. Ejecución del Agente

Inicia el agente interactivo ejecutando el archivo principal en tu terminal:

python agente_interactivo.py


El Agente iniciará un bucle de preguntas y respuestas hasta que escribas salir o exit.