import json
import httpx

IMPORT_URL = "http://localhost:8000/v1/session/import"
COMPLETIONS_URL = "http://localhost:8000/v1/chat/completions"
ADMIN_TOKEN = "test_admin_token_123"

def run_tests():
    # 1. Load active session JSON from session/session.json
    try:
        with open("session/session.json", "r", encoding="utf-8") as f:
            session_data = json.load(f)
    except Exception as e:
        print(f"Error: No se pudo leer session/session.json para la prueba: {e}")
        return

    print("Iniciando pruebas del endpoint de importación de sesión...")

    # 2. Test 1: Petición sin token (debe fallar con 401)
    print("\n[Test 1] Enviando importación sin token...")
    r = httpx.post(IMPORT_URL, json=session_data)
    print(f"Resultado: Código {r.status_code}")
    assert r.status_code == 401, "Error: Debería fallar sin token"

    # 3. Test 2: Petición con token inválido (debe fallar con 401)
    print("\n[Test 2] Enviando importación con token incorrecto...")
    headers = {"Authorization": "Bearer token_incorrecto_999"}
    r = httpx.post(IMPORT_URL, json=session_data, headers=headers)
    print(f"Resultado: Código {r.status_code}")
    assert r.status_code == 401, "Error: Debería fallar con token incorrecto"

    # 4. Test 3: Petición exitosa con token correcto (debe retornar 200)
    print("\n[Test 3] Enviando importación con token correcto...")
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    r = httpx.post(IMPORT_URL, json=session_data, headers=headers)
    print(f"Resultado: Código {r.status_code} -> {r.text}")
    assert r.status_code == 200, "Error: Debería ser exitoso"
    resp_data = r.json()
    assert resp_data["status"] == "ok", "Error: Respuesta incorrecta"

    # 5. Test 4: Verificar que el proxy sigue respondiendo tras la importación
    print("\n[Test 4] Verificando que el proxy sigue operativo...")
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Dime hola."}]
    }
    completion_headers = {"Authorization": "Bearer test_proxy_key_123"}
    r = httpx.post(COMPLETIONS_URL, json=payload, headers=completion_headers, timeout=30.0)
    print(f"Resultado: Código {r.status_code}")
    if r.status_code == 200:
        content = r.json()['choices'][0]['message']['content'].strip()
        safe_content = content.encode('ascii', errors='replace').decode('ascii')
        print(f"Respuesta del modelo: '{safe_content}'")
    assert r.status_code == 200, "Error: El proxy falló al responder tras la importación"

    print("\nSUCCESS: ¡TODAS LAS PRUEBAS DE IMPORTACIÓN PASARON CON ÉXITO!")

if __name__ == "__main__":
    run_tests()
