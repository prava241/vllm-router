# controller.py

import random, string
from models import *
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
        self.rolling_server_to_worker: float
        self.rolling_worker_to_server: float
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
        self.workers = {}
        self.last_worker_id = 0
        self.alphabet = string.ascii_letters + string.digits

    def register_user(self, priority):
        def generate_id():
            return ''.join([random.choice(self.alphabet) for _ in range(6)])
        user_id = generate_id()
        while user_id in self.users:
            user_id = generate_id
        self.users[user_id] = priority
        return user_id

    async def handle_request(
        self,
        user_id,
        request
    ):
        priority = self.users[user_id]
        worker = random.choice(
            list(self.workers.values())
        )
        
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{worker.address}/generate",
                json=request.model_dump()
            )

        # TODO: design change to allow for worker reassignment and aggregating results metrics:
        # assign a worker to the task
        # keep a set of ready results
        # keep polling, pop once you read your results
        
        # TODO: extract relevant metrics
        return r
    
    def delete_user(self, user_id):
        del self.users[user_id]
    
    def register_worker(self, info):
        self.last_worker_id += 1
        self.workers[self.last_worker_id](WorkerState(info))
        return self.last_worker_id
    
    def update_heartbeat(self, worker_id, info):
        pass

    def reassign(self, tasks):
        pass

    # TODO: run this in the background
    def heartbeat_loop(self):
        for worker_id, worker in self.workers:
            if time.time() - worker.last_heartbeat > 7:
                self.reassign(worker.active + list(worker.queue))
                # tell it to kill itself