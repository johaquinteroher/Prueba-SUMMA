#1. BASE: Usamos una imagen oficial de Python 3.10 optimizada para ser pequeña

FROM python:3.10-slim

#2. WORKDIR: Establecer el directorio de trabajo dentro del contenedor

WORKDIR /app

#3. DEPENDENCIAS: Copiar el archivo de requisitos e instalar las librerías

#La bandera --no-cache-dir reduce el tamaño final del contenedor

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#4. CÓDIGO: Copiar todos los archivos necesarios para la aplicación
#Esto incluye el código (main/model) y el modelo entrenado (PKL)

COPY main.py .
COPY model.py .
COPY ClasificacionAlphaBetha.pkl .

#5. PUERTO: Informar a Docker que la aplicación usa el puerto 8000

EXPOSE 8000

#6. COMANDO: Definir el comando que se ejecuta al iniciar el contenedor

#El host 0.0.0.0 permite que sea accesible desde fuera del contenedor

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]