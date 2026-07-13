import time
from queue import Queue
from typing import Any, Literal
from pydantic import BaseModel

class WorkerInfo(BaseModel):
    address: str
    gpu: str
    gpu_memory_total: int

class Heartbeat(BaseModel):
    worker_id: int

    gpu_utilization: float
    gpu_memory_free: int

    inference_latencies: list[float]
    tokens_per_second: list[float]

    uptime: float
    # controller can track RTT to worker, active requests, queue length

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Request(BaseModel):
    prompt: str

    system_prompt: str | None = None
    history: list[ChatMessage] = []

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95