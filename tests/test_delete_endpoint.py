import json
import time
from pathlib import Path
import httpx

COMPLETIONS_URL = "http://localhost:8000/v1/chat/completions"
DELETE_URL_TEMPLATE = "http://localhost:8000/v1/chat/session/{user_id}"
HEADERS = {"Authorization": "Bearer test_proxy_key_123"}
USER_ID = "test_delete_endpoint_user_999"

def run_tests():
    mapping_file = Path("session/user_threads.json")
    
    print("Iniciando prueba de eliminación por HTTP endpoint...")

    # 1. Crear un chat inicial
    print(f"\n[Test 1] Creando chat inicial para '{USER_ID}'...")
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hola, esto es una prueba para ser eliminada de inmediato."}],
        "user": USER_ID
    }
    
    r = httpx.post(COMPLETIONS_URL, json=payload, headers=HEADERS, timeout=60.0)
    print(f"Resultado: Código {r.status_code}")
    assert r.status_code == 200, f"Error al crear chat: {r.text}"
    
    # 2. Verificar que se mapeó en JSON
    time.sleep(1.0)
    assert mapping_file.exists(), "Error: El archivo de mapeo no existe."
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    assert USER_ID in mapping_data, "Error: El usuario no se guardó en el archivo de mapeo."
    print("SUCCESS: Chat guardado y mapeado correctamente.")

    # 3. Llamar al endpoint DELETE
    delete_url = DELETE_URL_TEMPLATE.format(user_id=USER_ID)
    print(f"\n[Test 2] Llamando a DELETE {delete_url}...")
    
    r_del = httpx.delete(delete_url, headers=HEADERS, timeout=30.0)
    print(f"Resultado: Código {r_del.status_code}")
    print(f"Respuesta del servidor: {r_del.text}")
    assert r_del.status_code == 200, f"Error al borrar sesión: {r_del.text}"
    
    # 4. Verificar que se borró del mapeo local
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_data_after = json.load(f)
    assert USER_ID not in mapping_data_after, "Error: El usuario aún existe en el archivo de mapeo local."
    print("SUCCESS: El usuario fue removido correctamente de session/user_threads.json.")
    
    print("\nSUCCESS: ¡EL ENDPOINT DE ELIMINACIÓN HTTP FUNCIONA Y FUE VERIFICADO CON ÉXITO!")

if __name__ == "__main__":
    run_tests()
