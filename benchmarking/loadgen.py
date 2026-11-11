import asyncio, random, sys, httpx

async def worker(c, stop):
    while not stop.is_set():
        body = {"prompt_tokens": random.choice([300, 800, 2000, 6000]), "max_tokens": random.choice([50, 120, 250])}
        try: await c.post("http://localhost:8000/v1/chat", json=body, timeout=60)
        except Exception: pass

async def main(concurrency: int, seconds: int):
    stop = asyncio.Event()
    async with httpx.AsyncClient() as c:
        tasks = [asyncio.create_task(worker(c, stop)) for _ in range(concurrency)]
        await asyncio.sleep(seconds); stop.set(); await asyncio.gather(*tasks)

asyncio.run(main(int(sys.argv[1]), int(sys.argv[2])))
