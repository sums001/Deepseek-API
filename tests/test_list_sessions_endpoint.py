import json
import time
from pathlib import Path
import httpx

LIST_URL = "http://localhost:8000/v1/chat/sessions"
COMPLETIONS_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Authorization": "Bearer test_proxy_key_123"}
USER_ID = "test_list_endpoint_user_777"

def run_tests():
    mapping_file = Path("session/user_threads.json")
    
    print("Iniciando prueba de listado de sesiones por HTTP endpoint...")

    # 1. Crear un chat temporal para asegurar que al menos una sesión tiene mapeo local
    print(f"\n[Test 1] Creando chat inicial para '{USER_ID}'...")
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hola, crea esta conversación para probar el listado."}],
        "user": USER_ID
    }
    r = httpx.post(COMPLETIONS_URL, json=payload, headers=HEADERS, timeout=60.0)
    print(f"Resultado: Código {r.status_code}")
    assert r.status_code == 200, f"Error al crear chat: {r.text}"
    
    # Esperar para guardar
    time.sleep(1.0)

    # 2. Obtener lista completa de sesiones
    print(f"\n[Test 2] Consultando lista de sesiones sin filtros...")
    r_list = httpx.get(LIST_URL, headers=HEADERS, timeout=30.0)
    print(f"Resultado: Código {r_list.status_code}")
    assert r_list.status_code == 200, f"Error al listar: {r_list.text}"
    
    sessions = r_list.json()
    print(f"Total de sesiones encontradas: {len(sessions)}")
    assert isinstance(sessions, list), "Error: La respuesta debe ser una lista."
    
    # 3. Verificar que al menos una sesión tenga nuestro user_id mapeado
    found_user = False
    for s in sessions:
        if s.get("user_id") == USER_ID:
            found_user = True
            print(f"SUCCESS: Encontrada la sesión asociada a '{USER_ID}' con ID: {s.get('id')} y título: '{s.get('title')}'")
            break
            
    assert found_user, f"Error: No se encontró ninguna sesión asociada a '{USER_ID}' en el listado."

    # 4. Probar filtro de fecha (ej: days=1, las últimas 24 horas)
    print(f"\n[Test 3] Consultando lista filtrando por days=1 (últimas 24 horas)...")
    r_today = httpx.get(f"{LIST_URL}?days=1", headers=HEADERS, timeout=30.0)
    print(f"Resultado: Código {r_today.status_code}")
    assert r_today.status_code == 200, f"Error al filtrar: {r_today.text}"
    
    today_sessions = r_today.json()
    print(f"Sesiones hoy: {len(today_sessions)}")
    
    # Verificar que el chat recién creado esté en la lista de hoy
    found_today = any(s.get("user_id") == USER_ID for s in today_sessions)
    assert found_today, "Error: El chat recién creado debería estar en la lista de hoy."
    print("SUCCESS: El chat recién creado figura correctamente en la lista de hoy.")

    # --- Limpieza del chat creado ---
    print(f"\nLimpiando chat creado para la prueba...")
    delete_url = f"http://localhost:8000/v1/chat/session/{USER_ID}"
    httpx.delete(delete_url, headers=HEADERS, timeout=30.0)

    print("\nSUCCESS: ¡EL ENDPOINT DE LISTADO Y FILTRO HTTP FUNCIONA Y FUE VERIFICADO CON ÉXITO!")

if __name__ == "__main__":
    run_tests()
