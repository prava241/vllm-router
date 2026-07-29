# models.py

import time
from queue import Queue
from typing import Any, Literal, List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class Heartbeat(BaseModel):
    worker_id: str

    mem_used: float
    mem_total: float
    gpu_util: float
    mem_util: float

class WorkerInfo(BaseModel):
    address: str
    heartbeat: Heartbeat

class UserRequest(BaseModel):
    request_id: str | None = None
    timestamp: float | None = None

    prompt_token_ids: list[int]
    prompt: str | None = None

    # prefix_hash: str | None

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    stop: list[str] | None = None

@dataclass(slots=True)
class GenerateRequest:
    request_id: str
    priority: Priority
    user_request: UserRequest
    retries: int = 0
    future: asyncio.Future = field(default_factory=asyncio.Future)
    enqueue_time: float = field(default_factory=time.time)
    scheduled_time: float | None = None
    dispatch_time: float | None = None

class GenerationMetrics(BaseModel):
    dispatch_latency: float
    timestamp: float
    ttft: float
    total_latency: float
    tps: float

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    num_cached_tokens: int | None = None

class GenerateResponse(BaseModel):
    request_id: str
    text: str
    finish_reason: str
    metrics: GenerationMetrics
