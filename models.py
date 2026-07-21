# models.py

import time
from queue import Queue
from typing import Any, Literal, List, Optional
from pydantic import BaseModel
from enum import Enum

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class Heartbeat(BaseModel):
    worker_id: int

    mem_used: float
    mem_total: float
    gpu_util: float
    mem_util: float

class WorkerInfo(BaseModel):
    address: str
    heartbeat: Heartbeat

class GenerateRequest(BaseModel):
    timestamp: float

    prompt_token_ids: list[int]
    prompt: str | None = None

    # prefix_hash: str | None

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    stop: list[str] | None = None

class GenerationMetrics(BaseModel):
    dispatch_time: float
    timestamp: float
    ttft: float
    total_latency: float
    tps: float

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class GenerateResponse(BaseModel):
    text: str
    finish_reason: str
    metrics: GenerationMetrics
