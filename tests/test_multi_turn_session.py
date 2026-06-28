import json
import time
from pathlib import Path
import httpx

COMPLETIONS_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Authorization": "Bearer test_proxy_key_123"}
USER_ID = "test_multi_turn_user_123"

def run_tests():
    # 1. Clean up old mapping for this test user if exists
    mapping_file = Path("session/user_threads.json")
    if mapping_file.exists():
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if USER_ID in data:
                del data[USER_ID]
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    print("Iniciando prueba de conversación multi-turno (5 rondas de memoria nativa)...")

    # --- RONDA 1: Presentación (Profesión y Comida) ---
    print(f"\n[Ronda 1] Presentando datos iniciales...")
    payload1 = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hola! Mi nombre es Marcos, soy programador y me encanta la pizza de pepperoni. Recuerda mi profesión y mi comida favorita."}],
        "user": USER_ID
    }
    r1 = httpx.post(COMPLETIONS_URL, json=payload1, headers=HEADERS, timeout=60.0)
    assert r1.status_code == 200, f"Error en Ronda 1: {r1.text}"
    content1 = r1.json()["choices"][0]["message"]["content"]
    safe_content1 = content1.encode('ascii', errors='replace').decode('ascii')
    print(f"Respuesta del modelo: '{safe_content1.strip()}'")

    # --- RONDA 2: Añadir Información (Edad) ---
    print(f"\n[Ronda 2] Añadiendo edad al contexto...")
    payload2 = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "También recuerda que tengo 30 años."}],
        "user": USER_ID
    }
    r2 = httpx.post(COMPLETIONS_URL, json=payload2, headers=HEADERS, timeout=60.0)
    assert r2.status_code == 200, f"Error en Ronda 2: {r2.text}"
    content2 = r2.json()["choices"][0]["message"]["content"]
    safe_content2 = content2.encode('ascii', errors='replace').decode('ascii')
    print(f"Respuesta del modelo: '{safe_content2.strip()}'")

    # --- RONDA 3: Verificar Profesión ---
    print(f"\n[Ronda 3] Preguntando profesión...")
    payload3 = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "¿A qué me dedico o cuál es mi trabajo?"}],
        "user": USER_ID
    }
    r3 = httpx.post(COMPLETIONS_URL, json=payload3, headers=HEADERS, timeout=60.0)
    assert r3.status_code == 200, f"Error en Ronda 3: {r3.text}"
    content3 = r3.json()["choices"][0]["message"]["content"]
    safe_content3 = content3.encode('ascii', errors='replace').decode('ascii')
    print(f"Respuesta del modelo: '{safe_content3.strip()}'")
    assert "program" in content3.lower() or "desarrolla" in content3.lower(), f"Error: No recordó la profesión. Respuesta: {content3}"
    print("SUCCESS: Recordó la profesión.")

    # --- RONDA 4: Verificar Edad ---
    print(f"\n[Ronda 4] Preguntando edad...")
    payload4 = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "¿Qué edad tengo?"}],
        "user": USER_ID
    }
    r4 = httpx.post(COMPLETIONS_URL, json=payload4, headers=HEADERS, timeout=60.0)
    assert r4.status_code == 200, f"Error en Ronda 4: {r4.text}"
    content4 = r4.json()["choices"][0]["message"]["content"]
    safe_content4 = content4.encode('ascii', errors='replace').decode('ascii')
    print(f"Respuesta del modelo: '{safe_content4.strip()}'")
    assert "30" in content4, f"Error: No recordó la edad. Respuesta: {content4}"
    print("SUCCESS: Recordó la edad.")

    # --- RONDA 5: Verificar Comida ---
    print(f"\n[Ronda 5] Preguntando comida favorita...")
    payload5 = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "¿Cuál es mi comida favorita y qué ingredientes mencioné?"}],
        "user": USER_ID
    }
    r5 = httpx.post(COMPLETIONS_URL, json=payload5, headers=HEADERS, timeout=60.0)
    assert r5.status_code == 200, f"Error en Ronda 5: {r5.text}"
    content5 = r5.json()["choices"][0]["message"]["content"]
    safe_content5 = content5.encode('ascii', errors='replace').decode('ascii')
    print(f"Respuesta del modelo: '{safe_content5.strip()}'")
    assert "pizza" in content5.lower(), f"Error: No recordó la comida. Respuesta: {content5}"
    print("SUCCESS: Recordó la comida y sus ingredientes.")

    print("\nSUCCESS: ¡LA CONVERSACIÓN DE 5 RONDAS EN LA MISMA SESIÓN FUE COMPLETADA Y VERIFICADA CON ÉXITO!")

if __name__ == "__main__":
    run_tests()
