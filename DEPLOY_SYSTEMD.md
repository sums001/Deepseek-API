# Guía de Despliegue con systemd para DeepSeek API Wrapper

Esta guía explica cómo configurar y ejecutar el proxy de DeepSeek API como un servicio en segundo plano usando **systemd** en un servidor Linux (ej. Ubuntu/Debian), bajo el directorio `/root/Deepseek-API/` y utilizando su entorno virtual.

---

## 📋 Requisitos Previos en el Servidor

1. **Clonar/Copiar el Repositorio:**
   Asegúrate de que los archivos estén en la ruta destino:
   ```bash
   ls -la /root/Deepseek-API/
   ```

2. **Crear y Preparar el Entorno Virtual:**
   Crea el entorno virtual `.venv` e instala las dependencias.
   ```bash
   cd /root/Deepseek-API
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Instalar Dependencias de Playwright (Muy Importante en Linux sin interfaz gráfica):**
   Para que Chromium (el navegador headless usado para resolver PoW y autenticar) funcione correctamente, debes instalar el navegador y las librerías del sistema necesarias:
   ```bash
   # Activar el entorno virtual si no lo está
   source .venv/bin/activate

   # Instalar Chromium
   playwright install chromium

   # Instalar dependencias del sistema operativo para Chromium (requiere sudo/root)
   playwright install-deps chromium
   ```

4. **Configurar el archivo `.env`:**
   Copia el archivo de ejemplo y edítalo:
   ```bash
   cp .env.example .env
   nano .env
   ```
   Asegúrate de configurar los siguientes valores mínimos para producción/headless:
   ```env
   HOST=127.0.0.1  # o 0.0.0.0 si quieres exponerlo directamente (se recomienda usar Nginx como proxy inverso)
   PORT=8000
   SERVER_INTERACTIVE_LOGIN=false # Evita que intente abrir un navegador en el servidor
   DEEPSEEK_PROXY_API_KEY=tu_api_key_segura_aqui
   PROXY_ADMIN_TOKEN=tu_token_de_administracion
   ```

5. **Transferir / Iniciar Sesión:**
   Dado que el servidor no tiene interfaz gráfica (`SERVER_INTERACTIVE_LOGIN=false`), no se abrirá el navegador para loguearte interactivamente. Tienes dos opciones para iniciar sesión:
   
   - **Opción A (Recomendada):** Iniciar sesión en tu PC local y exportar la sesión:
     1. En tu máquina local ejecuta: `python -m deepseek.auth --export`
     2. Esto generará un archivo (o un string JSON). Copia el contenido de la carpeta `session/` de tu máquina local y pégalo en `/root/Deepseek-API/session/` en el servidor.
   - **Opción B:** Usar el endpoint administrativo de importación una vez que el servidor esté corriendo, enviando la sesión mediante `/v1/session/import` utilizando tu `PROXY_ADMIN_TOKEN`.

---

## ⚙️ Configuración del Servicio systemd

El archivo [deepseek-api.service](file:///root/Deepseek-API/deepseek-api.service) ya está creado en la raíz del proyecto. Su contenido es el siguiente:

```ini
[Unit]
Description=Deepseek API Wrapper Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/Deepseek-API
# Ejecuta python directamente desde el entorno virtual (esto lo activa automáticamente)
ExecStart=/root/Deepseek-API/.venv/bin/python app.py
Restart=always
RestartSec=5

# Variables de entorno
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

> **Nota sobre el entorno virtual:** En systemd, ejecutar directamente el ejecutable de python dentro de la carpeta del entorno virtual (`/root/Deepseek-API/.venv/bin/python`) tiene el mismo efecto que activar el entorno virtual (`source .venv/bin/activate`), por lo que no es necesario envolverlo en scripts adicionales de activación.

### Pasos para Activar el Servicio:

1. **Copiar o enlazar el archivo de servicio a systemd:**
   ```bash
   ln -s /root/Deepseek-API/deepseek-api.service /etc/systemd/system/deepseek-api.service
   ```

2. **Recargar el daemon de systemd:**
   ```bash
   systemctl daemon-reload
   ```

3. **Habilitar el servicio para que inicie con el sistema:**
   ```bash
   systemctl enable deepseek-api
   ```

4. **Iniciar el servicio:**
   ```bash
   systemctl start deepseek-api
   ```

5. **Verificar el estado del servicio:**
   ```bash
   systemctl status deepseek-api
   ```

---

## 📊 Comandos Útiles de Mantenimiento

* **Ver los logs en tiempo real (Salida estándar y errores):**
  ```bash
  journalctl -u deepseek-api -f
  ```

* **Reiniciar el servicio (útil tras actualizar el código o `.env`):**
  ```bash
  systemctl restart deepseek-api
  ```

* **Detener el servicio:**
  ```bash
  systemctl stop deepseek-api
  ```
