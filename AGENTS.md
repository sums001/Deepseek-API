# Directrices de Desarrollo para DeepSeek API Wrapper

Este archivo contiene reglas y directrices para agentes de IA que trabajan en este repositorio.

## Descripción del Proyecto
Este proyecto es un proxy no oficial de la API de DeepSeek que convierte la interfaz de chat web gratuita (`chat.deepseek.com`) en una API compatible con OpenAI. Esto se logra mediante la automatización de navegadores (Playwright) para el inicio de sesión y la resolución en sandbox de los desafíos de prueba de trabajo (PoW) utilizando WebAssembly (`wasmtime`).

---

## 🚨 Reglas Críticas de Comportamiento

1. **Uso Responsable y Moderación**:
   - No elimines ni eludas el sistema de límite de tasa (`server/ratelimit.py`).
   - Las solicitudes concurrentes son soportadas gracias al diseño de WebAssembly concurrente (instanciación local de `wasmtime.Store` e `Instance` por petición con precopilado global de módulo y motor en `pow.py`). Evita reintroducir bloqueos de exclusión mutua global innecesarios.

2. **Mantenimiento del Entorno Sandbox (WASM/PoW)**:
   - El archivo `deepseek/sha3_wasm_bg.wasm` y su cargador `deepseek/pow.py` son fundamentales para resolver los desafíos de Proof-of-Work (PoW). Cualquier modificación en esta área debe ser minuciosamente probada.
   - Mantén la dependencia de `wasmtime` configurada de forma aislada y sin permisos de red/archivo dentro de su entorno.

3. **Manejo de Sesiones y Credenciales**:
   - El token de portador (`bearer token`) y las cookies se extraen del navegador persistente y se guardan localmente en `session/session.json`.
   - El mapeo de usuarios y hilos se almacena en `session/user_threads.json`.
   - Asegúrate de que todos los archivos dentro del directorio `session/` permanezcan ignorados por Git (`.gitignore`) para evitar filtraciones de credenciales e identificadores de usuario.
   - En entornos remotos/headless, se debe utilizar el flujo de importación/exportación de sesión (`python -m deepseek.auth --export` y `--import`) para sincronizar sesiones sin exponer las credenciales en Git.
   - En entornos remotos/headless, la variable `SERVER_INTERACTIVE_LOGIN` debe desactivarse de manera segura para retornar un error `503 Login Required` en lugar de intentar abrir un navegador interactivo.

4. **Compatibilidad con OpenAI**:
   - Al agregar nuevos endpoints o modificar `/v1/chat/completions`, se debe conservar la compatibilidad estricta con el SDK oficial de OpenAI.
   - El campo estándar `user` de OpenAI debe interceptarse y mapearse automáticamente a sesiones de DeepSeek si no se provee `conversation_id`.
   - Los parámetros no soportados por DeepSeek Web (como `temperature`, `top_p`, etc.) deben aceptarse silenciosamente para evitar errores en clientes existentes, a menos que sea estrictamente necesario lo contrario.

---

## Estructura del Código

- `deepseek/`: Contiene el cliente HTTP directo, el resolvedor de PoW y los flujos de autenticación automatizados con Playwright.
- `server/`: Contiene la lógica del servidor FastAPI, el formateador de respuestas al estilo OpenAI y la limitación de tasa.
- `examples/`: Código de demostración para el uso directo de la biblioteca de Python y para la integración del cliente de OpenAI.
