# server.py

import os
from fastapi import FastAPI
from src.controller import Controller
from pydantic import BaseModel
from src.models import *

from contextlib import asynccontextmanager

ROUTER_POLICY = os.environ.get("ROUTER_POLICY", "random")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"scheduling policy: {controller.policy_name}")
    controller.start()
    yield

app = FastAPI(lifespan=lifespan)
controller = Controller(policy=ROUTER_POLICY)

# Session Management, User Requests

@app.post("/sessions")
async def register_user(priority: Priority):
    return controller.register_user(priority)

@app.post("/sessions/{id}/request")
async def request(id, body: UserRequest):
    # this should return a response and metrics (latency, queue time, generation time)
    return await controller.handle_request(
        id,
        body
    )

@app.delete("/sessions/{id}")
async def delete_user(id):
    return controller.delete_user(id)

# Worker Management

@app.post("/workers/register")
async def register_worker(worker: WorkerInfo):
    # returns id
    return controller.register_worker(worker)

@app.post("/workers/heartbeat")
def heartbeat(worker_id: str, info: Heartbeat):
    controller.update_heartbeat(
        worker_id,
        info
    )

    return {
        "status": "ok"
    }