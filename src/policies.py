# policies.py

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.controller import WorkerState


class SchedulingPolicy(ABC):
    @abstractmethod
    def choose_worker(self, request, workers: dict[str, "WorkerState"]) -> "WorkerState":
        ...


class RandomPolicy(SchedulingPolicy):
    def choose_worker(self, request, workers):
        return random.choice(list(workers.values()))


class RoundRobinPolicy(SchedulingPolicy):
    def __init__(self):
        self._last_index = -1

    def choose_worker(self, request, workers):
        keys = sorted(workers.keys())
        self._last_index = (self._last_index + 1) % len(keys)
        return workers[keys[self._last_index]]


class LeastQueueDepthPolicy(SchedulingPolicy):
    def choose_worker(self, request, workers):
        def load(worker):
            return worker.queue.qsize() + len(worker.active)

        return min(workers.values(), key=load)


POLICIES: dict[str, type[SchedulingPolicy]] = {
    "random": RandomPolicy,
    "round_robin": RoundRobinPolicy,
    "least_queue_depth": LeastQueueDepthPolicy,
}
