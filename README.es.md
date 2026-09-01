# DeepSeek API: una API gratuita de LLM impulsada por DeepSeek

Usando tu propia cuenta de DeepSeek. Sin clave API, sin créditos, sin plan de pago: convierte el chat gratuito en [chat.deepseek.com](https://chat.deepseek.com) en una API que puedes llamar desde tu código.

Puedes usarlo de dos maneras:

🐍 **Como librería de Python:** simplemente llama a `client.chat("Hola")`. Soporta streaming y conversaciones de múltiples turnos.

🔌 **Como una API local compatible con OpenAI:** ejecuta un servidor en `http://localhost:8000/v1` que habla el formato de OpenAI, por lo que el SDK oficial de `openai` (y cualquier aplicación compatible con OpenAI) funciona como un reemplazo directo, usando `localhost` en lugar de OpenAI.

Inicias sesión una vez en el navegador con tu cuenta de DeepSeek; tu sesión se guarda y se renueva automáticamente después de eso.

*Proyecto no oficial. No afiliado ni respaldado por DeepSeek. Automatiza la experiencia web de consumo de DeepSeek para uso personal, así que úsalo de manera responsable y dentro de los términos de DeepSeek.*

## Tabla de contenidos

- [¿Por qué usar esto?](#por-qué-usar-esto)
- [Requisitos](#requisitos)
- [Configuración (2 minutos)](#configuración-2-minutos)
- [Uso 1: En Python (sin servidor)](#uso-1-en-python-sin-servidor)
- [Uso 2: Como un servidor compatible con OpenAI](#uso-2-como-un-servidor-compatible-con-openai)
- [Línea de comandos](#línea-de-comandos)
- [Verificación humana y prueba de trabajo (automático)](#verificación-humana-y-prueba-de-trabajo-automático)
- [Modelos, DeepThink y búsqueda web](#modelos-deepthink-y-búsqueda-web)
- [Concurrencia](#concurrencia)
- [Limitación de tasa (Rate limiting)](#limitación-de-tasa-rate-limiting)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Notas y limitaciones](#notas-y-limitaciones)
- [Licencia](#licencia)

## ¿Por qué usar esto?

- **Gratuito:** usa tu cuenta normal de DeepSeek con sesión iniciada, sin facturación de API.
- **Reemplazo directo de OpenAI:** apunta cualquier cliente de OpenAI a `localhost` y simplemente funciona.
- **Conjunto de herramientas completo de DeepSeek:** elige el modelo rápido o experto, y activa el razonamiento DeepThink y la búsqueda web por petición.
- **Streaming + conversaciones:** salida token por token y hilos de múltiples turnos dirigidos por `conversation_id`.

## Requisitos

- Python 3.9+
- Una cuenta de DeepSeek (la gratuita que usas para [chat.deepseek.com](https://chat.deepseek.com) es suficiente)
- Funciona en Windows, macOS y Linux

## Configuración (2 minutos)

```bash
# 1. Clonar el proyecto
git clone https://github.com/sums001/Deepseek-API.git
cd "Deepseek-API"
```

**2. Crear y activar un entorno virtual**

En macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

En Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
*En Windows, es posible que necesites permitir la ejecución de scripts una vez: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. En `cmd.exe` actívalo con `venv\Scripts\activate.bat` en su lugar.*

**3. Instalar dependencias e iniciar sesión**

```bash
# Instalar dependencias
pip install -r requirements.txt

# Instalar el navegador que Playwright necesita (una sola vez)
playwright install chromium

# Iniciar sesión una vez: se abre un navegador, inicia sesión en tu cuenta de DeepSeek
python -m deepseek.auth
```

La ventana de inicio de sesión se abre para que inicies sesión manualmente y resuelvas la verificación humana una vez. Después de eso, tu sesión (token de portador + cookies) se guarda en `session/` (ignorado por git, nunca compartido) y se reutiliza en cada ejecución; la sesión en caché se renueva automáticamente, por lo que tu primera petición funciona de inmediato.

*El servidor también puede abrir esta ventana por ti bajo demanda la primera vez que necesite una sesión, por lo que este paso es opcional para un uso local de un solo usuario.*

## Uso 1: En Python (sin servidor)

La forma más sencilla si tu código ya está en Python.

```python
from deepseek import DeepSeekClient

client = DeepSeekClient()                # carga tu sesión iniciada

# Obtener una respuesta completa
reply = client.chat("Saluda en una oración corta.")
print(reply.text)

# Continuar la MISMA conversación — pasa el id de vuelta
reply2 = client.chat("¿Y ahora en francés?", conversation_id=reply.conversation_id)
print(reply2.text)

# Transmitir la respuesta a medida que se escribe
for chunk in client.stream("Cuéntame un chiste corto"):
    print(chunk, end="", flush=True)
```

`chat()` devuelve el texto completo más un `conversation_id`; pasa ese id de vuelta para mantener el hilo, u omítelo para empezar de nuevo. `stream()` devuelve la respuesta pieza por pieza.

👉 **Más:** [examples/01_direct_chat.py](examples/01_direct_chat.py), [02_direct_conversation.py](examples/02_direct_conversation.py), [03_direct_stream.py](examples/03_direct_stream.py)

## Uso 2: Como un servidor compatible con OpenAI

Inicia un servidor local que habla la API de OpenAI, para que las herramientas y SDKs existentes de OpenAI funcionen sin cambios.

```bash
python app.py
# -> DeepSeek OpenAI-compatible API en http://127.0.0.1:8000
```

Luego apunta cualquier cliente de OpenAI a él (el SDK requiere la clave API pero se ignora):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "¡Hola!"}],
)
print(resp.choices[0].message.content)
```

O llámalo con HTTP simple / `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "¡Hola!"}]}'
```

### Endpoints

| Método | Ruta | Descripción |
| --- | --- | --- |
| POST | `/v1/chat/completions` | Chat (soporta `"stream": true`, más `"conversation_id"`, `"thinking"`, `"search"` opcionales) |
| GET | `/v1/models` | Lista los modelos disponibles |
| GET | `/healthz` | Comprobación de estado (exento de límite de tasa) |

Cambia la dirección con variables de entorno: `HOST=0.0.0.0 PORT=8080 python app.py`, o ejecuta `uvicorn server.api:app --host 0.0.0.0 --port 8080`.

👉 **Más:** [examples/04_server_http.py](examples/04_server_http.py), [05_server_stream.py](examples/05_server_stream.py), [06_server_openai_sdk.py](examples/06_server_openai_sdk.py)

## Línea de comandos

```bash
python -m deepseek.auth          # iniciar sesión y guardar la sesión
```

## Verificación humana y prueba de trabajo (automático)

El chat de DeepSeek está detrás de dos puertas, ambas manejadas por ti:

- **Verificación humana de AWS WAF:** el acceso necesita una sesión de navegador iniciada que haya superado la comprobación de "verificar que eres humano". `python -m deepseek.auth` abre un navegador real para que inicies sesión y lo resuelvas una vez; el token y las cookies resultantes se almacenan en caché en `session/` y se reutilizan en cada petición.
- **Prueba de trabajo (PoW):** cada completado está protegido por un desafío PoW. El puente lo resuelve ejecutando el propio módulo `sha3_wasm_bg.wasm` de DeepSeek — el mismo que carga el navegador — dentro de un sandbox de `wasmtime`, por lo que no tienes que hacer nada por tu parte.

Una sesión en caché se reutiliza durante ~6 horas y se renueva sin interfaz gráfica desde tu perfil de Chrome guardado cuando es posible; solo una expiración completa te devuelve al navegador.

## Modelos, DeepThink y búsqueda web

El nombre del `model` selecciona qué modelo responde. DeepThink y la búsqueda web no son modelos, son interruptores ortogonales que pasas por petición.

| Modelo | Modo DeepSeek | Notas |
| --- | --- | --- |
| `deepseek-chat` | Instant | Modelo rápido por defecto |
| `deepseek-expert` | Expert | Más fuerte, más lento |

Pasa `thinking: true` (razonamiento DeepThink) y/o `search: true` (búsqueda web) en el cuerpo de la petición — o a través de `extra_body` del SDK de OpenAI:

```python
resp = client.chat.completions.create(
    model="deepseek-expert",
    messages=[{"role": "user", "content": "¿Qué ha cambiado en las noticias hoy?"}],
    extra_body={"thinking": True, "search": True},
)
```

`conversation_id`, `thinking` y `search` son extras no pertenecientes a OpenAI. El modelo de un hilo se fija en la creación, por lo que `model` no se puede combinar con `conversation_id` al reanudar. Los nombres de modelos desconocidos devuelven un `404` (sin caída silenciosa). Ver [server/config.py](server/config.py).

## Concurrencia

El servidor conecta una única cuenta de DeepSeek con sesión iniciada detrás de un cliente compartido. El almacén `wasmtime` del solucionador PoW no es reentrante, por lo que las llamadas ascendentes se serializan: las peticiones HTTP paralelas hacen cola detrás de un bloqueo y se ejecutan una a la vez (ver [server/api.py](server/api.py)). Esto es intencionado: el rendimiento es secuencial, no paralelo. Mantén bajas las peticiones concurrentes en vuelo y, por favor, no satures tu cuenta.

## Limitación de tasa (Rate limiting)

Además de la serialización, el puente impone un límite de tasa autoimpuesto con un limitador de ventana deslizante sin dependencias ([server/ratelimit.py](server/ratelimit.py)): limita las peticiones aceptadas por IP de cliente y devuelve un `429` estándar + `Retry-After` cuando lo excedes. `/healthz` está exento.

| Variable de entorno | Por defecto | Significado |
| --- | --- | --- |
| `RATE_LIMIT_PER_MINUTE` | 30 | Peticiones/minuto aceptadas por IP de cliente |

```bash
RATE_LIMIT_PER_MINUTE=60 python app.py   # aumentarlo
```

En el lado del cliente, usa retroceso exponencial. Los `429` transitorios se limpian si reintentas con retrasos crecientes (ej. 1s, 2s, 4s). El SDK oficial de `openai` hace esto automáticamente y honra `Retry-After`; con HTTP simple, añade algunos reintentos tú mismo.

## Estructura del proyecto

| Ruta | Qué hace |
| --- | --- |
| [deepseek/](deepseek/) | La librería principal: `DeepSeekClient`, auth/inicio de sesión en navegador ([auth.py](deepseek/auth.py)), el controlador HTTP ([client.py](deepseek/client.py)) y el solucionador PoW ([pow.py](deepseek/pow.py)) |
| [server/](server/) | El servidor FastAPI compatible con OpenAI |
| [examples/](examples/) | Ejemplos ejecutables para cada función ([examples/README.md](examples/README.md)) |
| [app.py](app.py) | Inicia el servidor |

## Notas y limitaciones

- **Inicia sesión una vez, luego reutiliza.** La sesión en caché se renueva automáticamente; solo vuelves a iniciar sesión si expira por completo.
- **Sé razonable.** Por favor, úsalo con moderación y no hagas spam ni lo satures con peticiones masivas automatizadas.
- **Sin conteo real de tokens.** `usage` en las respuestas es una estimación aproximada de ~4 caracteres/token.
- **La mayoría de los parámetros de OpenAI se aceptan pero se ignoran** (`temperature`, `top_p`, `max_tokens`); solo `model`, `messages`, `stream`, `conversation_id`, `thinking` y `search` hacen algo.
- **El soporte para visión está pendiente.** Requiere lógica de subida de imágenes que aún no está implementada.
- **Tu sesión es privada.** Todo en `session/` (cookies + token) se queda en tu máquina y es ignorado por git.

## Licencia

Publicado bajo la [Licencia MIT](LICENSE). Como este es un proyecto no oficial, sigues siendo responsable de cumplir con los términos de servicio de DeepSeek.
