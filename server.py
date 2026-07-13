from fastapi import FastAPI
from controller import Controller
from pydantic import BaseModel
from models import *

app = FastAPI()
controller = Controller()

# Session Management, User Requests

@app.post("/sessions")
async def create_session():
    return controller.create_session()

@app.post("/sessions/{id}/request")
async def request(id, body):
    # this should return a response and metrics (latency, queue time, generation time)
    return await controller.handle_request(
        id,
        body
    )

@app.delete("/sessions/{id}")
async def end_session(id):
    return controller.end_session(id)

# Worker Management

@app.post("/workers/register")
async def register_worker(worker: WorkerInfo):
    # returns id
    return controller.register_worker(worker)

# TODO: does heartbeat need to be async?
@app.post("/workers/heartbeat")
async def heartbeat(worker_id: str, info: Heartbeat):
    controller.update_heartbeat(
        worker_id,
        info
    )

    return {
        "status": "ok"
    }