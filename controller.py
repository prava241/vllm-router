import random, string
from models import *
import httpx

class WorkerState:
    def __init__(self, info: WorkerInfo) -> None:
        self.address: str = info.address
        self.active: list[Any] = []
        self.queue: Queue[Any] = Queue()
        self.last_heartbeat: float = time.time()
        self.worker_cache: dict[str, Any] = {}
        self.gpu = info.gpu
        self.gpu_memory_total = info.gpu_memory_total
        self.utilization
        self.latency

    def record_heartbeat(self, heartbeat: Heartbeat) -> None:
        self.last_heartbeat = time.time()
        self.utilization
        self.latency

    @property
    def is_alive(self, timeout: float = 30.0) -> bool:
        return (time.time() - self.last_heartbeat) < timeout

class Controller:
    def __init__(self):
        self.sessions = {}
        self.workers = []
        self.last_worker_id = 0
        self.alphabet = string.ascii_letters + string.digits
    
    def register_worker(self, info):
        # TODO: create new worker
        self.workers.append(WorkerState(info))
        self.last_worker_id += 1
        return self.last_worker_id

    def create_session(self):
        def generate_id():
            return ''.join([random.choice(self.alphabet) for _ in range(6)])
        session_id = generate_id()
        while session_id in self.sessions:
            session_id = generate_id
        self.sessions[session_id] = []
        return session_id


    async def handle_request(
        self,
        session_id,
        request
    ):
        # TODO: choose a worker
        worker = self.workers[0]
        
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{worker.address}/generate",
                json=request.model_dump()
            )


    def end_session(self, session_id):
        del self.sessions[session_id]