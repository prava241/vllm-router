# controller.py

import random, string
from src.models import *
from src.policies import POLICIES
import httpx
import asyncio
from dataclasses import dataclass, field
import time
import uuid

MAX_RETRIES = 3
def ema(old: float | None, new: float, alpha: float = 0.2) -> float:
    if old is None:
        return new
    return alpha * new + (1 - alpha) * old

class WorkerState:
    def __init__(self, controller, info: WorkerInfo) -> None:
        self.controller = controller
        self.address: str = info.address

        self.queue = asyncio.Queue()
        self.active = set()
        self.last_heartbeat: float = time.time()

        self.gpu_util: float
        self.mem_remaining: float
        self.mem_util: float
        
        # self.rolling_queue_length: float
        self.rolling_dispatch_latency: float | None = None
        self.rolling_return_latency: float | None = None
        self.rolling_RTT: float | None = None
        self.rolling_ttft: float | None = None
        self.rolling_tps: float | None = None

        self.dispatch_task = asyncio.create_task(
            self.dispatch_loop()
        )

    def record_heartbeat(self, heartbeat: Heartbeat) -> None:
        self.last_heartbeat = time.time()
        self.gpu_util = heartbeat.gpu_util
        self.mem_remaining = heartbeat.mem_total - heartbeat.mem_used
        self.mem_util = heartbeat.mem_util

    def record_completion(
        self,
        response: GenerateResponse,
        rtt: float
    ):
        metrics : GenerationMetrics = response.metrics
        return_time = time.time()
        return_latency = return_time - metrics.timestamp

        self.rolling_dispatch_latency = ema(
            self.rolling_dispatch_latency,
            metrics.dispatch_latency,
        )

        self.rolling_return_latency = ema(
            self.rolling_return_latency,
            return_latency,
        )

        self.rolling_RTT = ema(
            self.rolling_RTT,
            rtt,
        )

        self.rolling_ttft = ema(
            self.rolling_ttft,
            metrics.ttft,
        )

        self.rolling_tps = ema(
            self.rolling_tps,
            metrics.tps,
        )

    async def dispatch_loop(self):
        async with httpx.AsyncClient(timeout=None) as client:
            while True:
                request = await self.queue.get()
                request.dispatch_time = time.time()
                self.active.add(request)
                try:
                    request.user_request.request_id = request.request_id
                    request.user_request.timestamp = time.time()
                    start = time.perf_counter()
                    r = await client.post(
                        f"{self.address}/generate",
                        json=request.user_request.model_dump(),
                    )
                    end = time.perf_counter()
                    r.raise_for_status()
                    response = GenerateResponse.model_validate(
                        r.json()
                    )
                    self.record_completion(
                        response,
                        rtt=end - start,
                    )
                    request.future.set_result(response)
                except Exception as e:
                    await self.controller.handle_failure(
                        request,
                        e,
                    )
                finally:
                    self.active.discard(request)
                    self.queue.task_done()

    @property
    def is_alive(self, timeout: float = 30.0) -> bool:
        return (time.time() - self.last_heartbeat) < timeout

class Controller:
    def __init__(self, policy: str = "random"):
        if policy not in POLICIES:
            raise ValueError(
                f"Unknown policy '{policy}', available: {list(POLICIES)}"
            )
        self.policy_name = policy
        self._policy = POLICIES[policy]()

        self.users = {}
        self.workers: dict[str, WorkerState] = {}
        self.last_worker_id = 0
        self.alphabet = string.ascii_letters + string.digits
        self.queue = asyncio.Queue() # change this to priority queue
        self.scheduling_task = None
        self.heartbeat_task = None

    def start(self):
        self.scheduling_task = asyncio.create_task(
            self.scheduler_loop()
        )
        self.heartbeat_task = asyncio.create_task(
            self.heartbeat_loop()
        )

    def register_user(self, priority):
        def generate_uid():
            return "u_" + ''.join([random.choice(self.alphabet) for _ in range(6)])
        user_id = generate_uid()
        while user_id in self.users:
            user_id = generate_uid()
        self.users[user_id] = priority
        return user_id

    async def handle_request(
        self, 
        user_id,
        user_request
    ):
        request = GenerateRequest(
            request_id=str(uuid.uuid4()),
            priority = self.users[user_id],
            user_request = user_request
        )

        await self.queue.put(request)
        response = await request.future

        queue_latency = (
            request.scheduled_time
            - request.enqueue_time
        )

        return response

    async def scheduler_loop(self):
        while True:
            request = await self.queue.get()
            worker = self.choose_worker(request)
            request.scheduled_time = time.time()
    
            await worker.queue.put(request)
            self.queue.task_done()

            # Could do priority checks, rate limits,
            # resource allocation, etc. here

    def choose_worker(self, request):
        return self._policy.choose_worker(request, self.workers)

    async def handle_failure(
        self,
        request,
        error,
    ):
        if request.retries < MAX_RETRIES:
            request.retries += 1
            await self.queue.put(request)
            return
        request.future.set_exception(error)

    def delete_user(self, user_id):
        del self.users[user_id]
    
    def register_worker(self, info):
        def generate_wid():
            return "w_" + ''.join([random.choice(self.alphabet) for _ in range(6)])
        worker_id = generate_wid()
        while worker_id in self.workers:
            worker_id = generate_wid()
        self.workers[worker_id] = (WorkerState(self, info))
        return worker_id
    
    def update_heartbeat(self, worker_id, heartbeat):
        self.workers[worker_id].record_heartbeat(heartbeat)

    async def heartbeat_loop(self):
        while True:
            dead_workers = []
            for worker_id, worker in self.workers.items():
                if not worker.is_alive:
                    dead_workers.append(worker_id)

            for worker_id in dead_workers:
                worker = self.workers.pop(worker_id)
                print(f"Worker {worker_id} timed out.")
                await self.reassign_worker(worker)

            await asyncio.sleep(5)

    async def reassign_worker(
        self,
        worker: WorkerState,
    ):
        worker.dispatch_task.cancel()

        while not worker.queue.empty():
            request = await worker.queue.get()
            await self.queue.put(request)
            worker.queue.task_done()

        for request in list(worker.active):
            request.retries += 1
            await self.queue.put(request)