# controller.py

import random, string
from src.models import *
import httpx

class WorkerState:
    def __init__(self, info: WorkerInfo) -> None:
        self.address: str = info.address

        self.active: list[Any] = []
        self.queue: Queue[Any] = Queue()
        self.last_heartbeat: float = time.time()

        self.gpu_util: float
        self.mem_remaining: float
        self.mem_util: float
        
        # self.rolling_queue_length: float
        self.rolling_dispatch_latency: float
        self.rolling_return_latency: float
        self.rolling_RTT: float
        self.rolling_ttft: float
        self.rolling_tps: float

    def record_heartbeat(self, heartbeat: Heartbeat) -> None:
        self.last_heartbeat = time.time()
        self.gpu_util = heartbeat.gpu_util
        self.mem_remaining = heartbeat.mem_total - heartbeat.mem_used
        self.mem_util = heartbeat.mem_util

    def record_completion(self, response: GenerateResponse):
        pass

    @property
    def is_alive(self, timeout: float = 30.0) -> bool:
        return (time.time() - self.last_heartbeat) < timeout

class Controller:
    def __init__(self):
        self.users = {}
        self.workers: dict[str, WorkerState] = {}
        self.last_worker_id = 0
        self.alphabet = string.ascii_letters + string.digits

    def register_user(self, priority):
        def generate_uid():
            return "u_" + ''.join([random.choice(self.alphabet) for _ in range(6)])
        user_id = generate_uid()
        while user_id in self.users:
            user_id = generate_uid()
        self.users[user_id] = priority
        return user_id
    
    async def dispatch_request(self, worker_id, request):
        worker = self.workers[worker_id]
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{worker.address}/generate",
                json=request.model_dump()
            )
        worker.record_completion(response)
        return response

    async def handle_request(
        self,
        user_id,
        request
    ):
        priority = self.users[user_id]
        worker_id = random.choice(
            list(self.workers.keys())
        )

        return await self.dispatch_request(worker_id, request)
    
    def delete_user(self, user_id):
        del self.users[user_id]
    
    def register_worker(self, info):
        def generate_wid():
            return "w_" + ''.join([random.choice(self.alphabet) for _ in range(6)])
        worker_id = generate_wid()
        while worker_id in self.users:
            worker_id = generate_wid()
        self.workers[worker_id] = (WorkerState(info))
        return worker_id
    
    def update_heartbeat(self, worker_id, heartbeat):
        self.workers[worker_id].record_heartbeat(heartbeat)

    def reassign(self, tasks):
        pass

    # TODO: run this in the background
    def heartbeat_loop(self):
        for worker_id, worker in self.workers.items():
            if not worker.is_alive:
                # reassign all tasks
                # kill worker
                # remove it from registry
                pass