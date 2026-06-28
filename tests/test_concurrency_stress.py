import asyncio
import time
import httpx

URL = "http://localhost:8000/v1/chat/completions"
NUM_REQUESTS = 5

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "Genera una sola palabra graciosa."}
    ],
    "stream": False
}

async def make_request(client: httpx.AsyncClient, request_id: int):
    start = time.time()
    try:
        response = await client.post(URL, json=payload, timeout=60.0)
        elapsed = time.time() - start
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            print(f"[Req {request_id}] Completo en {elapsed:.2f}s -> '{content}'")
            return elapsed
        else:
            print(f"[Req {request_id}] Falló con estado {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"[Req {request_id}] Error: {e}")
        return None

async def run_concurrently():
    print(f"\n--- Probando {NUM_REQUESTS} peticiones concurrentes ---")
    start_total = time.time()
    async with httpx.AsyncClient() as client:
        tasks = [make_request(client, i) for i in range(NUM_REQUESTS)]
        results = await asyncio.gather(*tasks)
    total_time = time.time() - start_total
    
    successful = [r for r in results if r is not None]
    if successful:
        avg_time = sum(successful) / len(successful)
        print(f"Total tiempo concurrente: {total_time:.2f}s")
        print(f"Promedio de tiempo por petición individual: {avg_time:.2f}s")
    return total_time

async def run_sequentially():
    print(f"\n--- Probando {NUM_REQUESTS} peticiones secuenciales ---")
    start_total = time.time()
    results = []
    async with httpx.AsyncClient() as client:
        for i in range(NUM_REQUESTS):
            res = await make_request(client, i)
            results.append(res)
    total_time = time.time() - start_total
    
    successful = [r for r in results if r is not None]
    if successful:
        avg_time = sum(successful) / len(successful)
        print(f"Total tiempo secuencial: {total_time:.2f}s")
        print(f"Promedio de tiempo por petición individual: {avg_time:.2f}s")
    return total_time

async def main():
    print("Iniciando pruebas de concurrencia y estrés contra el proxy...")
    # Ejecutar secuencial primero para establecer la línea de base lenta
    seq_time = await run_sequentially()
    # Ejecutar concurrente después para ver el paralelismo
    con_time = await run_concurrently()
    
    print("\n================ RESULTADOS ================")
    print(f"Tiempo Total Secuencial: {seq_time:.2f}s")
    print(f"Tiempo Total Concurrente: {con_time:.2f}s")
    if con_time > 0:
        speedup = seq_time / con_time
        print(f"Factor de aceleración (Speedup): {speedup:.2f}x")
        if speedup > 1.5:
            print("SUCCESS: Las peticiones se ejecutaron concurrentemente.")
        else:
            print("WARNING: El speedup es bajo. Esto puede ocurrir si el backend de DeepSeek limita las peticiones concurrentes para la misma sesión o la conexión a internet es inestable.")
    else:
        print("No se pudo calcular el speedup debido a fallos en las peticiones.")

if __name__ == "__main__":
    asyncio.run(main())
