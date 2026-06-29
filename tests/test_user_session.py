import json
import os
import time
from pathlib import Path
import httpx

COMPLETIONS_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Authorization": "Bearer test_proxy_key_123"}
USER_ID = "test_user_marcos_99"
DIFFERENT_USER_ID = "test_user_isolated_88"

def run_tests():
    # 1. Clean up old mapping for this test user if exists
    mapping_file = Path("session/user_threads.json")
    if mapping_file.exists():
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if USER_ID in data:
                del data[USER_ID]
            if DIFFERENT_USER_ID in data:
                del data[DIFFERENT_USER_ID]
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    print("Iniciando pruebas de mapeo de sesión y memoria nativa...")

    # --- TEST 1: Crear una nueva conversación con un usuario específico ---
    print(f"\n[Test 1] Iniciando chat para '{USER_ID}' y revelando secreto...")
    payload1 = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hola, mi nombre es Marcos y mi perro se llama Max. Recuerda el nombre de mi perro."}],
        "user": USER_ID
    }
    
    r1 = httpx.post(COMPLETIONS_URL, json=payload1, headers=HEADERS, timeout=60.0)
    print(f"Resultado: Código {r1.status_code}")
    assert r1.status_code == 200, f"Error en Test 1: {r1.text}"
    content1 = r1.json()["choices"][0]["message"]["content"]
    safe_content1 = content1.encode('ascii', errors='replace').decode('ascii')
    print(f"Respuesta del modelo: '{safe_content1.strip()}'")

    # --- TEST 2: Verificar persistencia en archivo JSON ---
    print("\n[Test 2] Verificando persistencia en session/user_threads.json...")
    time.sleep(1.0) # Esperar a que se guarde el archivo
    assert mapping_file.exists(), "Error: El archivo de mapeo no existe."
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    
    assert USER_ID in mapping_data, f"Error: No se encontró la sesión para el usuario '{USER_ID}'"
    mapped_cid = mapping_data[USER_ID]
    print(f"SUCCESS: El usuario '{USER_ID}' está correctamente mapeado a la sesión '{mapped_cid}'")

    # --- TEST 3: Preguntar al mismo usuario (debería recordar el secreto por memoria nativa de DeepSeek) ---
    print(f"\n[Test 3] Consultando a '{USER_ID}' (sin pasar historial local, solo la nueva pregunta)...")
    payload2 = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "¿Cómo se llama mi perro?"}],
        "user": USER_ID
    }
    
    r2 = httpx.post(COMPLETIONS_URL, json=payload2, headers=HEADERS, timeout=60.0)
    print(f"Resultado: Código {r2.status_code}")
    assert r2.status_code == 200, f"Error en Test 3: {r2.text}"
    content2 = r2.json()["choices"][0]["message"]["content"]
    safe_content2 = content2.encode('ascii', errors='replace').decode('ascii')
    print(f"Respuesta del modelo: '{safe_content2.strip()}'")
    
    # Verificar que el modelo mencione a "Max"
    assert "max" in content2.lower(), f"Error: El modelo no recordó el nombre del perro. Respuesta: '{content2}'"
    print("SUCCESS: ¡El modelo recordó el nombre mediante memoria nativa de DeepSeek!")

    # --- TEST 4: Aislamiento de sesiones (otro usuario no debería conocer el secreto del primero) ---
    print(f"\n[Test 4] Consultando a un usuario diferente '{DIFFERENT_USER_ID}' sobre el perro...")
    payload3 = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "¿Cómo se llama mi perro?"}],
        "user": DIFFERENT_USER_ID
    }
    
    r3 = httpx.post(COMPLETIONS_URL, json=payload3, headers=HEADERS, timeout=60.0)
    print(f"Resultado: Código {r3.status_code}")
    assert r3.status_code == 200, f"Error en Test 4: {r3.text}"
    content3 = r3.json()["choices"][0]["message"]["content"]
    safe_content3 = content3.encode('ascii', errors='replace').decode('ascii')
    print(f"Respuesta del modelo: '{safe_content3.strip()}'")
    
    # El usuario diferente no debe conocer el nombre "Max"
    assert "max" not in content3.lower(), f"Error: El usuario aislado '{DIFFERENT_USER_ID}' conoció el secreto del primer usuario. Respuesta: '{content3}'"
    print(f"SUCCESS: Aislamiento verificado. El usuario '{DIFFERENT_USER_ID}' no tiene acceso al contexto de '{USER_ID}'")

    print("\nSUCCESS: ¡TODAS LAS PRUEBAS DE MEMORIA Y SESIÓN NATIVA PASARON CON ÉXITO!")

if __name__ == "__main__":
    run_tests()
