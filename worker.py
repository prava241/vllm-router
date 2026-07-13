# worker.py

import argparse
from fastapi import FastAPI, BackgroundTasks
import httpx
import asyncio
from models import *

parser = argparse.ArgumentParser()
parser.add_argument("--controller-url")
args = parser.parse_args()
CONTROLLER_URL = args.controller_url

app = FastAPI()

WORKER_ID = "generated-at-registry"
WORKER_ADDRESS = "generated-at-startup" # TODO
MODEL = None

@app.on_event("startup")
async def startup():
    # connect to Colab GPU
    # get gpu, gpu_memory_total
    # load vLLM
    # get cloudflare tunnel
    # load vLLM
    global MODEL, WORKER_ADDRESS

    MODEL = load_vllm()
    asyncio.create_task(heartbeat_loop())

    WORKER_ADDRESS = "" #TODO

    await register(gpu, gpu_memory_total)

async def register(gpu, gpu_memory_total):
    global WORKER_ID
    worker_payload = WorkerInfo(
        address=WORKER_ADDRESS, 
        gpu=gpu # TODO,
        gpu_memory_total=gpu_memory_total
    )
    async with httpx.AsyncClient() as client:
        WORKER_ID = await client.post(
            f"{CONTROLLER_URL}/workers/register",
            json=worker_payload.model_dump()
        )

async def heartbeat_loop():
    while True:

        async with httpx.AsyncClient() as client:
            await client.post(
                f"{CONTROLLER_URL}/workers/heartbeat",
                json={
                    "worker_id": WORKER_ID,
                    "gpu_util": get_gpu_utilization(),
                    "active_requests": active_requests
                }
            )

        await asyncio.sleep(5)


@app.post("/generate")
async def generate(request: Request):
    messages = []

    if request.system_prompt:
        messages.append({
            "role": "system",
            "content": request.system_prompt,
        })

    if request.history:
        messages.extend(request.history)

    messages.append({
        "role": "user",
        "content": request.prompt,
    })

    result = await MODEL.generate(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    return result


@app.get("/health")
async def health():
    return {
        "worker_id": WORKER_ID,
        "status": "ok"
    }

# TODO
async def cleanup():
    pass

@app.post("/shutdown")
async def shutdown():
    await cleanup()
    background_tasks.add_task(exit_process)
    return "worker terminated"