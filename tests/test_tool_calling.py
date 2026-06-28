import json
import httpx

URL = "http://localhost:8000/v1/chat/completions"

# Definir una herramienta ficticia para consultar el clima o cotizaciones
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_dolar_rate",
            "description": "Obtiene la cotización actual del dólar en Venezuela.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "¿A cuánto está la cotización del dólar en Venezuela actualmente? Usa la herramienta."}
    ],
    "tools": tools,
    "thinking": True
}

print("Enviando petición con herramienta 'get_dolar_rate' al servidor...")
try:
    with httpx.Client() as client:
        response = client.post(URL, json=payload, timeout=120.0)
        print(f"Estado de la respuesta: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("\nJSON de respuesta del Servidor Proxy:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # Verificar si hay una llamada a la herramienta
            message = data["choices"][0]["message"]
            if "tool_calls" in message:
                print("\n¡ÉXITO! Se detectó una llamada a función (tool_calls):")
                for tc in message["tool_calls"]:
                    print(f"  - ID: {tc['id']}")
                    print(f"  - Función: {tc['function']['name']}")
                    print(f"  - Argumentos: {tc['function']['arguments']}")
            else:
                print("\nEl modelo no solicitó la herramienta. Respuesta de texto:")
                print(message["content"])
        else:
            print(f"Error del servidor: {response.text}")
except Exception as e:
    print(f"Error de conexión: {e}")
