Proyecto: API de Clasificación Alpha/Betha (Prueba Técnica)

Este proyecto implementa un servicio RESTful para la clasificación de clientes en las categorías Alpha o Betha a través de una API construida con FastAPI. El objetivo es exponer el modelo de Machine Learning entrenado (Punto 2) como un microservicio listo para el consumo.

Estructura del Proyecto

El repositorio contiene todos los archivos necesarios para el entrenamiento, despliegue y prueba:

Archivo

Función dentro del Proyecto

main.py
Punto de Entrada de la API. Contiene la lógica de FastAPI, carga el modelo y define el endpoint /predict.

model.py
Clase ClassifierModel que gestiona la carga del modelo .pkl y realiza las predicciones (.predict_proba).

ClasificacionAlphaBetha.pkl
Artefacto. El modelo de Scikit-learn entrenado (output del Punto 2).

to_predict.csv
Datos de entrada para la prueba de integració

client.py
Script de Python para automatizar las peticiones POST y realizar la prueba de integración 

requirements.txt
Listado preciso de todas las dependencias de Python y sus versiones.

Dockerfile
(Opcional - Punto 5) Instrucciones para empaquetar la aplicación en un contenedor Docker.

1. Requisitos e Instalación

Requisito Mínimo: Tener Python 3.8+ instalado.

Clonar o Descargar el repositorio completo.

Navegar a la carpeta raíz del proyecto en la terminal.

Crear un Ambiente Virtual (Altamente recomendado para aislar dependencias):

python -m venv venv
.\venv\Scripts\activate  # Para Windows
source venv/bin/activate # Para Linux/macOS


Instalar Dependencias (Usando el requirements.txt generado):
pip install -r requirements.txt


2. Ejecución de la API (Punto 3 - Servidor)
La API debe iniciarse en una terminal y permanecer activa para recibir solicitudes.
Asegúrate de que tu ambiente virtual está activo.
Ejecuta el servidor Uvicorn (se reinicia automáticamente ante cambios en el código):
uvicorn main:app --reload
Verificación de Despliegue:
La API estará activa en: http://127.0.0.1:8000


3. Prueba de Integración (Punto 4 - Cliente)
Para verificar que la API funciona correctamente, se utiliza el script client.py que lee los datos de to_predict.csv y hace las peticiones.
Abre una SEGUNDA TERMINAL (manteniendo el servidor corriendo en la primera).
Asegúrate de que estás en el ambiente virtual y en la carpeta del proyecto.
Ejecuta el script de prueba: python client.py


Output Esperado:
El cliente mostrará la predicción (Alpha/Betha) y la probabilidad para cada uno de los 3 registros.
Finalmente, generará el archivo results_ClasificacionAlphaBetha.csv.
Con esto, has completado todos los requisitos de despliegue y documentación. ¡Tu proyecto está listo para ser entregado!