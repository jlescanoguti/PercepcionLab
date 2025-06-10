from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import face_recognition
import sqlite3
import json
import numpy as np
from numpy.linalg import norm

app = FastAPI()

# --- CONFIGURACIÓN ---
RUTA_ROSTROS = "rostros"
UMBRAL_SIMILITUD = 0.45

# --- FUNCIONES DE UTILIDAD ---

def calcular_distancia(v1, v2):
    return norm(np.array(v1) - np.array(v2))

def insertar_persona_si_nueva(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        print("❌ La imagen no existe.")
        return "error", "La imagen no existe."

    imagen = face_recognition.load_image_file(ruta_imagen)
    ubicaciones = face_recognition.face_locations(imagen)
    
    if len(ubicaciones) != 1:
        return "error", f"La imagen debe contener 1 rostro. Se encontraron: {len(ubicaciones)}."

    embedding_nuevo = face_recognition.face_encodings(imagen, known_face_locations=ubicaciones)[0]
    embedding_lista = embedding_nuevo.tolist()

    archivo = os.path.basename(ruta_imagen)
    nombre_base = os.path.splitext(archivo)[0]
    nombre = ''.join([c if c.isalpha() else ' ' for c in nombre_base])
    nombre = ' '.join(nombre.split())

    conn = sqlite3.connect("alumnos.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre, imagen, embedding FROM persona")
    registros = cursor.fetchall()

    for id_, nombre_bd, imagen_bd, emb_texto in registros:
        emb_bd = json.loads(emb_texto)
        distancia = calcular_distancia(embedding_lista, emb_bd)
        if distancia < UMBRAL_SIMILITUD:
            conn.close()
            return "duplicado", f"'{nombre}' ya se parece a '{nombre_bd}' (distancia = {distancia:.4f})"

    cursor.execute("INSERT INTO persona (nombre, imagen, embedding) VALUES (?, ?, ?)",
                   (nombre, archivo, json.dumps(embedding_lista)))
    conn.commit()
    conn.close()

    return "ok", f"{nombre} fue insertado correctamente."

# --- ENDPOINT PRINCIPAL ---

@app.post("/subir_imagen")
async def subir_imagen(file: UploadFile = File(...)):
    try:
        # Guardar la imagen temporalmente en la carpeta rostros
        ruta_destino = os.path.join(RUTA_ROSTROS, file.filename)
        with open(ruta_destino, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        estado, mensaje = insertar_persona_si_nueva(ruta_destino)

        if estado == "ok":
            return JSONResponse(status_code=200, content={"resultado": mensaje})
        elif estado == "duplicado":
            return JSONResponse(status_code=409, content={"resultado": mensaje})
        else:
            return JSONResponse(status_code=400, content={"error": mensaje})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
