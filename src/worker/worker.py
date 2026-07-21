# worker.py

import argparse
from fastapi import FastAPI, BackgroundTasks
import httpx
import asyncio
from src.models import *
from src.worker.vllm_engine import (
    VLLMModel,
    GenerateRequest,
)
import pynvml
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--controller-url")
args = parser.parse_args()
CONTROLLER_URL = args.controller_url

WORKER_ID = ""
WORKER_ADDRESS = None
MODEL = VLLMModel(
    "meta-llama/Llama-3.1-8B-Instruct"
)
HANDLE = None

TUNNEL_PATTERN = re.compile(
    r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com"
)
CLOUDFLARED_PROCESS = None

MISSED_HEARTBEATS = 0
MAX_MISSED_HEARTBEATS = 3

app = FastAPI()
def get_worker_metrics():
    mem = pynvml.nvmlDeviceGetMemoryInfo(HANDLE)
    util = pynvml.nvmlDeviceGetUtilizationRates(HANDLE)
    return Heartbeat(
        worker_id=WORKER_ID,
        mem_used=mem.used,
        mem_total=mem.total,
        gpu_util=util.gpu,
        mem_util=util.mem
    )

async def start_cloudflared(port: int = 8000) -> str:
    """Starts a Cloudflare tunnel and returns the public URL."""

    global CLOUDFLARED_PROCESS

    CLOUDFLARED_PROCESS = await asyncio.create_subprocess_exec(
        "./cloudflared",
        "tunnel",
        "--url",
        f"http://localhost:{port}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    while True:
        line = await CLOUDFLARED_PROCESS.stdout.readline()

        if not line:
            raise RuntimeError("cloudflared exited before producing a tunnel URL")

        line = line.decode().rstrip()
        print(line)

        match = TUNNEL_PATTERN.search(line)
        if match:
            return match.group(0)

@app.on_event("startup")
async def startup():
    global HANDLE, WORKER_ADDRESS

    pynvml.nvmlInit()
    HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)

    await MODEL.load()

    WORKER_ADDRESS = await start_cloudflared()

    await register(get_worker_metrics())

    asyncio.create_task(heartbeat_loop())

async def register(worker_metrics):
    global WORKER_ID
    worker_payload = WorkerInfo(
        address=WORKER_ADDRESS, 
        worker_metrics=worker_metrics
    )
    async with httpx.AsyncClient() as client:
        WORKER_ID = await client.post(
            f"{CONTROLLER_URL}/workers/register",
            json=worker_payload.model_dump()
        )

async def heartbeat_loop():
    global MISSED_HEARTBEATS

    while True:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(
                    f"{CONTROLLER_URL}/workers/heartbeat",
                    json={
                        "worker_id": WORKER_ID,
                        "heartbeat": get_worker_metrics().model_dump(),
                    },
                )

            r.raise_for_status()

            body = r.json()

            if body["status"] == "shutdown":
                print("Controller requested shutdown.")
                await cleanup()
                raise SystemExit

            MISSED_HEARTBEATS = 0

        except Exception as e:
            MISSED_HEARTBEATS += 1

            print(
                f"Heartbeat failed ({MISSED_HEARTBEATS}/"
                f"{MAX_MISSED_HEARTBEATS}): {e}"
            )

            if MISSED_HEARTBEATS >= MAX_MISSED_HEARTBEATS:
                print("Lost contact with controller.")
                await cleanup()
                raise SystemExit

        await asyncio.sleep(5)

@app.post("/generate")
async def generate(request: GenerateRequest):
    dispatch_latency = time.time() - request.timestamp
    result = await MODEL.generate(request, dispatch_latency)
    return result

async def cleanup():
    global CLOUDFLARED_PROCESS

    if CLOUDFLARED_PROCESS is not None:
        CLOUDFLARED_PROCESS.terminate()
        await CLOUDFLARED_PROCESS.wait()

    pynvml.nvmlShutdown()