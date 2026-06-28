# Integración con n8n (AI Agent & OpenAI Chat Model)

Este proxy es compatible con los nodos de Inteligencia Artificial de **n8n** (como el nodo **AI Agent** conectado a un **OpenAI Chat Model**). Debido a que n8n utiliza LangChain internamente, hay algunas configuraciones específicas que debes realizar para evitar errores de conexión.

---

## Configuración paso a paso en n8n

### 1. Nodos en el Flujo
1. Añade un nodo **AI Agent** a tu flujo.
2. Conecta un nodo **OpenAI Chat Model** a la entrada del modelo del agente.

### 2. Configurar la Credencial (OpenAI API)
En la configuración del nodo **OpenAI Chat Model**, crea una nueva credencial de tipo **OpenAI API** con los siguientes valores:
*   **API Key:** `unused` (o cualquier texto de relleno; es requerido por n8n pero ignorado por el servidor).
*   **Base URL Override (URL base de la API):** Haz clic en *Add Option* -> *Base URL* y escribe la dirección IP de tu servidor (ej. tu máquina Proxmox o localhost):
    ```text
    http://<TU_IP_DE_PROXMOX>:8000/v1
    ```
    *Nota: Asegúrate de incluir el `/v1` al final.*

### 3. Ajustes Avanzados del Nodo (Importante 🚨)
Para evitar que LangChain o n8n devuelvan un error `404 status code (no body)` debido a validaciones internas de respuesta del modelo, debes aplicar lo siguiente en la configuración del nodo de OpenAI:
*   **Desactivar "Use Response API":** Apaga la opción de respuesta de API avanzada si te da un error de respuesta del modelo. Esto hace que n8n confíe en el formato OpenAI simplificado devuelto por el proxy.

### 4. Selección de Modelos (DeepThink y Búsqueda Web)
Dado que n8n no permite enviar parámetros personalizados en el cuerpo del JSON (como `extra_body`), el servidor proxy mapea automáticamente las opciones avanzadas de DeepSeek mediante **modelos virtuales**. 

Puedes seleccionar o escribir los siguientes nombres de modelos en n8n:

| Nombre del Modelo en n8n | Modelo de DeepSeek | DeepThink (Razonamiento) | Búsqueda Web |
| --- | --- | :---: | :---: |
| `deepseek-chat` | Instant (Rápido) | ❌ | ❌ |
| `deepseek-chat-thinking` | Instant (Rápido) |  | ❌ |
| `deepseek-chat-search` | Instant (Rápido) | ❌ |  |
| `deepseek-chat-thinking-search` | Instant (Rápido) |  |  |
| `deepseek-expert` | Expert (Lento) | ❌ | ❌ |
| `deepseek-expert-thinking` | Expert (Lento) |  | ❌ |
| `deepseek-expert-search` | Expert (Lento) | ❌ |  |
| `deepseek-expert-thinking-search` | Expert (Lento) |  |  |

---

## Soporte de Herramientas (Tool/Function Calling)

El proxy soporta la ejecución de herramientas (**Tool/Function Calling**) desde n8n de forma nativa. 

Cuando conectas herramientas (por ejemplo, nodos de *Wikipedia*, *Google Search*, o *Custom HTTP Requests*) a tu nodo de **AI Agent** en n8n, n8n enviará las definiciones de estas herramientas al proxy bajo el parámetro `tools`.

El proxy emula esta funcionalidad automáticamente:
1. Convierte las definiciones de tus herramientas de n8n en una **instrucción de sistema** estructurada que le indica a DeepSeek cómo y cuándo responder en un formato JSON específico.
2. Si DeepSeek decide usar una herramienta, responde con dicho JSON.
3. El proxy intercepta esa respuesta, la convierte en un evento `tool_calls` estándar de OpenAI, y se la devuelve a n8n.
4. n8n ejecuta la herramienta localmente y envía de vuelta el resultado al proxy para que el agente termine de responder.

Esto funciona de manera completamente transparente y te permite crear agentes autónomos complejos en n8n usando este proxy.

---

## Comportamiento del Servidor y Compatibilidad

*   **Fallback automático:** Si n8n (o cualquier otra herramienta) envía un modelo que no se encuentra en la lista (como el modelo por defecto `gpt-3.5-turbo` o `gpt-4o`), el servidor aplicará un **fallback automático** a `deepseek-chat` para evitar que la automatización se detenga con un error 404.
*   **Comportamiento original mantenido:** Fuera de n8n y otros entornos rígidos, el funcionamiento original del servidor permanece completamente intacto. Las peticiones directas de código, SDKs oficiales de OpenAI que usen `extra_body` para pasar `"thinking": true` o `"search": true`, y cualquier otra conexión estándar seguirán funcionando exactamente como antes. Los sufijos virtuales son una capa de compatibilidad exclusivamente añadida para resolver las limitaciones de n8n.
