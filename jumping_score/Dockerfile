FROM mybase:latest

WORKDIR /app
COPY server.py scripts/ /app/

# para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir -r requirements.txt

EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]


## uvicorn: Se encarga de levantar tu aplicación Python y servirla por HTTP

## server:app:  server es el nombre del archivo Python server.py (sin el .py).
# app → es la variable dentro de ese archivo que contiene tu aplicación FastAPI (ej. app = FastAPI(...)).

## --host", "0.0.0.0: --host define en qué interfaz de red escuchar.
# Por defecto Uvicorn escucha en 127.0.0.1 (solo accesible desde dentro del contenedor).
# Al poner 0.0.0.0 → significa “escucha en todas las interfaces disponibles”.

## --port", "8000: El puerto dentro del contenedor en el que la app escucha.
#Aquí Uvicorn queda escuchando en 0.0.0.0:8000.
#Este número debe coincidir con el que uses en docker-compose.yml (parte de la derecha en ports: xxxx:8000).

## "--workers", "1": Número de procesos trabajadores que va a lanzar Uvicorn.
# 1 → un solo proceso (suficiente para desarrollo o cosas simples).
# En producción puedes aumentar (ej. --workers 4) para aprovechar CPUs y manejar más peticiones en paralelo.