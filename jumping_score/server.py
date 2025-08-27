from typing import List
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
from detection_jump import is_jumping
from calculadora import calcular_puntuacion
from fastapi.responses import JSONResponse

app = FastAPI(title="Score Calculator", version="1.0")


# Recibimos la llamada del servicio de tracking para procesar las personas que estan
# saltando
class ScoreRequest(BaseModel):
    data: List[List[float]]


class JumpingRequest(BaseModel):
    keypoints: List[List[float]]

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/isjumping")
def is_jumping_endpoint(payload: JumpingRequest):
    result, _ = is_jumping(payload.keypoints, separar_saltos=False)
    return {"is_jumping": bool(result)}

@app.post("/score")
def total_score(payload: ScoreRequest):

    score = calcular_puntuacion(payload.data)
    output = {
        "score": list(score)
    }
    try:
        return JSONResponse(output)
    
    except Exception as e:
        # Nunca devuelvas objetos raros en errores
        raise HTTPException(status_code=400, detail=str(e))
